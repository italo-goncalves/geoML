# geoML - machine learning models for geospatial data
# Copyright (C) 2020  Ítalo Gomes Gonçalves
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
import numpy as np
# import numpy as np
from scipy.linalg import helmert as _helmert
import copy as _copy

import geoml.warping as _warp
import geoml.parameter as _gpr
import geoml.tftools as _tftools
import geoml.interpolation as _gint

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp

_tfd = _tfp.distributions

_ROOTS_8 = _tf.constant(dtype=_tf.float64, value=[
    3.811869902073221168547189e-1,
    1.157193712446780194720766,
    1.981656756695842925854631,
    2.930637420257244019223503
])
_ROOTS_8 = _tf.concat([-_ROOTS_8[::-1], _ROOTS_8], axis=0)

_WEIGHTS_8 = _tf.constant(dtype=_tf.float64, value=[
    6.611470125582412910303848e-1,
    2.078023258148918795432488e-1,
    1.707798300741347545620225e-2,
    1.996040722113676192060810e-4
])
_WEIGHTS_8 = _tf.concat([_WEIGHTS_8[::-1], _WEIGHTS_8], axis=0)
_WEIGHTS_8 = _WEIGHTS_8 / _tf.reduce_sum(_WEIGHTS_8)

_ROOTS_64 = _tf.constant(dtype=_tf.float64, value=[
    1.383022449870097241150498e-1,
    4.149888241210786845769291e-1,
    6.919223058100445772682193e-1,
    9.692694230711780167435415e-1,
    1.247200156943117940693565,
    1.525889140209863662948970,
    1.805517171465544918908774,
    2.086272879881762020832563,
    2.368354588632401404111511,
    2.651972435430635011005458,
    2.937350823004621809685339,
    3.224731291992035725848171,
    3.514375935740906211539951,
    3.806571513945360461165972,
    4.101634474566656714970981,
    4.399917168228137647767933,
    4.701815647407499816097538,
    5.007779602198768196443703,
    5.318325224633270857323650,
    5.634052164349972147249920,
    5.955666326799486045344567,
    6.284011228774828235418093,
    6.620112262636027379036660,
    6.965241120551107529242642,
    7.321013032780949201189569,
    7.689540164040496828447804,
    8.073687285010225225858791,
    8.477529083379863090564166,
    8.907249099964769757295973,
    9.373159549646721162545652,
    9.895287586829539021204461,
    1.052612316796054588332683e1
])
_ROOTS_64 = _tf.concat([-_ROOTS_64[::-1], _ROOTS_64], axis=0)

_WEIGHTS_64 = _tf.constant(dtype=_tf.float64, value=[
    2.713774249413039779455939e-1,
    2.329947860626780466505551e-1,
    1.716858423490837020007199e-1,
    1.084983493061868406330207e-1,
    5.873998196409943454968617e-2,
    2.720312895368891845383354e-2,
    1.075604050987913704946467e-2,
    3.622586978534458760667954e-3,
    1.036329099507577663456693e-3,
    2.509838985130624860823502e-4,
    5.125929135786274660821669e-5,
    8.788499230850359181443633e-6,
    1.258340251031184576157783e-6,
    1.495532936727247061102391e-7,
    1.465125316476109354926553e-8,
    1.173616742321549343542451e-9,
    7.615217250145451353314936e-11,
    3.959177766947723927236259e-12,
    1.628340730709720362084230e-13,
    5.218623726590847522957562e-15,
    1.280093391322438041639503e-16,
    2.351884710675819116957565e-18,
    3.152254566503781416121198e-20,
    2.982862784279851154478560e-22,
    1.911706883300642829958367e-24,
    7.861797788925910369099620e-27,
    1.929103595464966850301878e-29,
    2.549660899112999256604646e-32,
    1.557390624629763802300262e-35,
    3.421138011255740504327060e-39,
    1.679747990108159218666209e-43,
    5.535706535856942820575202e-49
])
_WEIGHTS_64 = _tf.concat([_WEIGHTS_64[::-1], _WEIGHTS_64], axis=0)
_WEIGHTS_64 = _WEIGHTS_64 / _tf.reduce_sum(_WEIGHTS_64)


class _Likelihood(_gpr.Parametric):
    def __init__(self, size, use_monte_carlo=False):
        super().__init__()
        self._size = size
        self._use_monte_carlo = use_monte_carlo

    @property
    def size(self):
        return self._size

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        raise NotImplementedError

    def predict(self, mu, var, sims, explained_var, include_noise=True, *args, **kwargs):
        raise NotImplementedError

    def log_lik_from_samples(self, samples, y, has_value, *args, **kwargs):
        raise NotImplementedError

    def predict_from_samples(self, samples):
        raise NotImplementedError

    def initialize(self, y):
        pass


class _ContinuousLikelihood(_Likelihood):
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        """
        Initializer for continuous likelihoods.

        Parameters
        ----------
        warping : geoml.warping.Warping
            A Warping object that normalizes the data values.
        use_monte_carlo : bool
            Whether to use Monte Carlo samples in training, instead of the
            probability density function. Used mainly to maintain consistency
            while using other likelihood for another variable, which does not
            have an analytical form.
        """
        super().__init__(1, use_monte_carlo)
        self.warping = self._register(warping)
        self._spline = _gint.MonotonicCubicSpline()

    def initialize(self, y):
        self.warping.initialize(y)

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y_warped = self.warping.forward(y)
        y_derivative = self.warping.derivative(y)
        log_derivative = _tf.math.log(y_derivative)

        if self._use_monte_carlo:
            distribution = self._make_distribution(samples[:, 0, :])

            log_density = distribution.log_prob(y_warped)
            log_density = _tf.math.reduce_mean(
                log_density, axis=1, keepdims=True)

        else:
            vals = _tf.expand_dims(_ROOTS_64, axis=0)
            vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
            w = _tf.expand_dims(_WEIGHTS_64, axis=0)

            distribution = self._make_distribution(vals)

            log_density = distribution.log_prob(y_warped)
            log_density = _tf.reduce_sum(log_density * w, axis=1,
                                         keepdims=True)

        lik = _tf.reduce_sum((log_density + log_derivative) * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var, *args, include_noise=True, quantiles=None,
                probabilities=None, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        # w = _tf.expand_dims(_WEIGHTS, axis=0)

        # distribution = self._make_distribution(vals)
        distribution = _tfd.MixtureSameFamily(
            _tfd.Categorical(probs=_WEIGHTS_64),
            self._make_distribution(vals)
        )

        lik_var = distribution.variance()
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        # weights = weights**2

        sims = sims[:, 0, :]
        if include_noise:
            s = _tf.shape(sims)
            sims = sims + self.white_noise(s, seed=1234)

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": self.warping.backward(sims[:, 0, :]),
               "weights": _tf.squeeze(weights),
               }

        def prob_fn(q):
            q = _tf.expand_dims(q, 0)
            p = distribution.cdf(_tf.squeeze(self.warping.forward(q)))
            # p = _tf.reduce_sum(p * w, axis=1)
            return p

        if quantiles is not None:
            prob = _tf.map_fn(prob_fn, quantiles)
            prob = _tf.transpose(prob)

            # single point case
            prob = _tf.cond(
                _tf.less(_tf.rank(prob), 2),
                lambda: _tf.expand_dims(prob, 0),
                lambda: prob)

            # out["probabilities"] = _tf.squeeze(prob)
            out["probabilities"] = prob

        # def quant_fn(p):
        #     # p = _tf.expand_dims(p, 0)
        #     q = self.warping.backward(distribution.quantile(p))
        #     q = _tf.reduce_sum(q * w, axis=1)
        #     return q

        if probabilities is not None:
            warped_vals = self.warping.backward(vals)
            prob_vals = _tf.map_fn(lambda x: distribution.cdf(x),
                                   _tf.transpose(vals))

            n_data = _tf.shape(warped_vals)[0]
            quant = self._spline.interpolate(
                prob_vals,
                _tf.transpose(warped_vals),
                _tf.tile(probabilities[:, None], [1, n_data])
            )

            # quant = _tf.map_fn(quant_fn, probabilities)
            quant = _tf.transpose(quant)

            # # single point case
            # quant = _tf.cond(
            #     _tf.less(_tf.rank(quant), 2),
            #     lambda: _tf.expand_dims(quant, 0),
            #     lambda: quant)

            # out["quantiles"] = _tf.squeeze(quant)
            out["quantiles"] = quant

        return out

    def white_noise(self, shape, seed):
        n_data = shape[0]
        n_samples = shape[1]
        dist = self._make_distribution(_tf.zeros(n_data, _tf.float64))
        sample = dist.sample(n_samples, seed=seed)
        return _tf.transpose(sample)

    def _make_distribution(self, *args, **kwargs):
        raise NotImplementedError


class Gaussian(_ContinuousLikelihood):
    """
    Gaussian likelihood.

    Equivalent to a squared error model. The latent variable maps to the mean,
    while the noise variance is a parameter.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("noise", _gpr.PositiveParameter(0.1, 1e-12, 10))

    def _make_distribution(self, loc):
        return _tfd.Normal(loc, _tf.sqrt(self.parameters["noise"].get_value()))


class Laplace(_ContinuousLikelihood):
    """
    Laplace likelihood.

    Equivalent to a linear error model. The latent variable maps to the mean,
    while the distribution's scale factor is a parameter.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("scale", _gpr.PositiveParameter(0.1, 1e-9, 10))

    def _make_distribution(self, loc):
        return _tfd.Laplace(loc, self.parameters["scale"].get_value())


class Gamma(_ContinuousLikelihood):
    """
    Gamma likelihood.

    Used for strictly positive variables. The latent variable is shifted by a
    parameter and then mapped to the distribution's shape. The rate parameter
    is fixed at 1.0.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("mean_alpha", _gpr.RealParameter(0, -3, 3))

    def _make_distribution(self, loc):
        mean_alpha = self.parameters["mean_alpha"].get_value()
        return _tfd.Gamma(_tf.exp(loc + mean_alpha) + 0.01,
                          _tf.constant(1.0, _tf.float64))


class Beta(_ContinuousLikelihood):
    def __init__(self, concentration=1, use_monte_carlo=False):
        super().__init__(use_monte_carlo=use_monte_carlo)
        self._add_parameter("concentration", _gpr.PositiveParameter(
            concentration, 1e-3, 10, fixed=False))

    def _make_distribution(self, loc):
        loc = _tf.nn.sigmoid(loc)
        loc = _tf.maximum(
            _tf.constant(1e-6, _tf.float64),
            _tf.minimum(loc, _tf.constant(1 - 1e-6, _tf.float64)))
        concentration = self.parameters["concentration"].get_value()
        alpha = loc * concentration
        beta = (1 - loc) * concentration
        return _tfd.Beta(alpha, beta)


class StudentT(_ContinuousLikelihood):
    """
    Student-T likelihood.

    A heavy-tailed distribution. The latent variable maps to the mean,
    while the scale and degrees of freedom are parameters.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("scale", _gpr.PositiveParameter(0.1, 1e-9, 10))
        self._add_parameter("df", _gpr.PositiveParameter(5.0, 2.01, 50.0))

    def _make_distribution(self, loc):
        return _tfd.StudentT(
            df=self.parameters["df"].get_value(),
            loc=loc,
            scale=self.parameters["scale"].get_value())


class EpsilonInsensitive(_ContinuousLikelihood):
    """
    Epsilon-insensitive likelihood.

    Similar to the Laplace likelihood, with an addition `epsilon` parameter,
    below which error are not penalized. Can be used to obtain a model similar
    to the Support Vector Machine.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("epsilon", _gpr.PositiveParameter(0.001, 1e-9, 10))
        self._add_parameter("c_rate", _gpr.PositiveParameter(1, 1e-3, 1e3))

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y_warped = self.warping.forward(y)
        y_derivative = self.warping.derivative(y)
        log_derivative = _tf.math.log(y_derivative)

        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        if self._use_monte_carlo:
            samples = samples[:, 0, :]

            y_centered = _tf.math.abs(y_warped - samples)

            log_density = _tf.where(
                _tf.less_equal(y_centered, epsilon),
                _tf.zeros_like(y_centered),
                - c_rate * (y_centered - epsilon)
            )
            log_density = log_density - _tf.math.log(
                2 * (epsilon + 1 / c_rate))
            log_density = _tf.reduce_mean(log_density, axis=1, keepdims=True)

        else:
            vals = _tf.expand_dims(_ROOTS_64, axis=0)
            vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
            w = _tf.expand_dims(_WEIGHTS_64, axis=0)

            y_centered = _tf.math.abs(y_warped - vals)

            log_density = _tf.where(
                _tf.less_equal(y_centered, epsilon),
                _tf.zeros_like(y_centered),
                - c_rate * (y_centered - epsilon)
            )
            log_density = log_density - _tf.math.log(
                2 * (epsilon + 1 / c_rate))
            log_density = _tf.math.reduce_logsumexp(
                log_density + _tf.math.log(w), axis=1, keepdims=True)

        lik = _tf.reduce_sum(
            (log_density + log_derivative) * has_value)

        return lik

    def cdf(self, x):
        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        val_1 = _tf.math.exp(c_rate * (x + epsilon)) / c_rate
        val_2 = 1 / c_rate + x - epsilon
        val_3 = 2 * epsilon + 2 / c_rate \
                * (1 - 0.5 * _tf.math.exp(- c_rate * (x - epsilon)))

        prob = _tf.where(_tf.greater(x, -epsilon), val_2, val_1)
        prob = _tf.where(_tf.greater(x, epsilon), val_3, prob)

        prob = prob * 0.5 / (epsilon + 1 / c_rate)
        return prob

    def inv_cdf(self, p):
        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        area = 2 * (epsilon + 1 / c_rate)
        p_1 = (1 / c_rate) / area
        p_2 = p_1 + 2 * epsilon / area

        val_1 = _tf.math.log(area * c_rate * p) / c_rate - epsilon
        val_2 = area * p - 1 / c_rate - epsilon
        val_3 = epsilon - _tf.math.log(2 - c_rate * (area * p - 2 * epsilon)) / c_rate

        x = _tf.where(_tf.greater(p, p_1), val_2, val_1)
        x = _tf.where(_tf.greater(p, p_2), val_3, x)
        return x

    def variance(self):
        e = self.parameters["epsilon"].get_value()
        c = self.parameters["c_rate"].get_value()

        n = 3 * c * e * (c * e + 2) + 6 + (c * e) ** 3
        d = 3 * c ** 2 * (c * e + 1)
        return n / d

    def white_noise(self, shape, seed):
        rnd = _tf.random.uniform(shape, minval=0.0, maxval=1.0, seed=seed, dtype=_tf.float64)
        sample = self.inv_cdf(rnd)
        return sample

    def predict(self, mu, var, sims, explained_var, *args, quantiles=None,
                probabilities=None, **kwargs):
        lik_var = self.variance()
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        weights = weights ** 2

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": self.warping.backward(sims[:, 0, :]),
               "weights": weights
               }

        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        def prob_fn(q):
            q = _tf.expand_dims(q, 0)
            p = self.cdf(_tf.squeeze(self.warping.forward(q)) - vals)
            p = _tf.reduce_sum(p * w, axis=1)
            return p

        if quantiles is not None:
            prob = _tf.map_fn(prob_fn, quantiles)
            prob = _tf.transpose(prob)

            # single point case
            # prob = _tf.cond(
            #     _tf.less(_tf.rank(prob), 2),
            #     lambda: _tf.expand_dims(prob, 0),
            #     lambda: prob)

            out["probabilities"] = prob

        # def quant_fn(p):
        #     p = _tf.expand_dims(p, 0)
        #     q = distribution.quantile(_tf.squeeze(self.warping.backward(p)))
        #     q = _tf.reduce_sum(q * w, axis=1)
        #     return q
        #
        if probabilities is not None:
            def prob_fn_2(q):
                q = _tf.expand_dims(q, 1)
                p = self.cdf(q - vals)
                p = _tf.reduce_sum(p * w, axis=1)
                return p

            prob_vals = _tf.map_fn(prob_fn_2, _tf.transpose(vals))

            n_data = _tf.shape(vals)[0]
            quant = self._spline.interpolate(
                prob_vals,
                _tf.transpose(self.warping.backward(vals)),
                _tf.tile(probabilities[:, None], [1, n_data])
            )

            # quant = _tf.map_fn(quant_fn, probabilities)
            quant = _tf.transpose(quant)

            # single point case
            # quant = _tf.cond(
            #     _tf.less(_tf.rank(quant), 2),
            #     lambda: _tf.expand_dims(quant, 0),
            #     lambda: quant)

            out["quantiles"] = quant

        return out


class Huber(_ContinuousLikelihood):
    """
    Huber likelihood.

    Based on the Huber loss.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._add_parameter("threshold", _gpr.PositiveParameter(3, 1e-2, 10))
        self._add_parameter("std", _gpr.PositiveParameter(0.1, 1e-3, 1))

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y_warped = self.warping.forward(y)
        y_derivative = self.warping.derivative(y)
        log_derivative = _tf.math.log(y_derivative)

        thr = self.parameters["threshold"].get_value()
        std = self.parameters["std"].get_value()

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr**2) / thr)

        if self._use_monte_carlo:
            samples = samples[:, 0, :]

            y_centered = _tf.math.abs((y_warped - samples) / std)

            log_density = _tf.where(
                _tf.less_equal(y_centered, thr),
                - 0.5 * y_centered**2,
                - thr * (y_centered - 0.5*thr)
            )
            log_density = log_density - _tf.math.log(norm)
            log_density = _tf.reduce_mean(log_density, axis=1, keepdims=True)

        else:
            vals = _tf.expand_dims(_ROOTS_64, axis=0)
            vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
            w = _tf.expand_dims(_WEIGHTS_64, axis=0)

            y_centered = _tf.math.abs((y_warped - vals) / std)

            log_density = _tf.where(
                _tf.less_equal(y_centered, thr),
                - 0.5 * y_centered ** 2,
                - thr * (y_centered - 0.5 * thr)
            )
            log_density = log_density - _tf.math.log(norm)
            log_density = _tf.math.reduce_logsumexp(
                log_density + _tf.math.log(w), axis=1, keepdims=True)

        lik = _tf.reduce_sum(
            (log_density + log_derivative) * has_value)

        return lik

    def cdf(self, x):
        thr = self.parameters["threshold"].get_value()
        std = self.parameters["std"].get_value()

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / thr)

        x = x / std

        val_1 = _tf.math.exp(thr * x + 0.5 * thr ** 2) / thr
        val_2 = _tf.math.exp(-0.5 * thr ** 2) / thr \
                + _np.sqrt(_np.pi / 2) * (_tf.math.erf(thr / _np.sqrt(2))
                                          + _tf.math.erf(x / _np.sqrt(2)))
        val_3 = _tf.math.exp(-0.5 * thr ** 2) / thr \
                + 2 * _np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2)) \
                + (_tf.math.exp(-0.5*thr**2) -
                   _tf.math.exp(0.5*thr**2 - thr*x)) / thr

        prob = _tf.where(_tf.greater(x, -thr), val_2, val_1)
        prob = _tf.where(_tf.greater(x, thr), val_3, prob)

        prob = prob / norm
        return prob

    def variance(self):
        thr = self.parameters["threshold"].get_value()
        std = self.parameters["std"].get_value()

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / thr)

        n1 = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                  - thr * _tf.math.exp(-0.5 * thr ** 2))
        n2 = 2 * _tf.math.exp(-0.5 * thr ** 2) \
             * (thr + 2/thr + 2 / thr**3)
        return (n1 + n2) / norm * std**2

    def predict(self, mu, var, sims, explained_var, *args, quantiles=None,
                probabilities=None, **kwargs):
        lik_var = self.variance()
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        weights = weights ** 2

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": self.warping.backward(sims[:, 0, :]),
               "weights": weights
               }

        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        def prob_fn(q):
            q = _tf.expand_dims(q, 0)
            p = self.cdf(_tf.squeeze(self.warping.forward(q)) - vals)
            p = _tf.reduce_sum(p * w, axis=1)
            return p

        if quantiles is not None:
            prob = _tf.map_fn(prob_fn, quantiles)
            prob = _tf.transpose(prob)

            # single point case
            # prob = _tf.cond(
            #     _tf.less(_tf.rank(prob), 2),
            #     lambda: _tf.expand_dims(prob, 0),
            #     lambda: prob)

            out["probabilities"] = prob

        # def quant_fn(p):
        #     p = _tf.expand_dims(p, 0)
        #     q = distribution.quantile(_tf.squeeze(self.warping.backward(p)))
        #     q = _tf.reduce_sum(q * w, axis=1)
        #     return q
        #
        if probabilities is not None:
            def prob_fn_2(q):
                q = _tf.expand_dims(q, 1)
                p = self.cdf(q - vals)
                p = _tf.reduce_sum(p * w, axis=1)
                return p

            prob_vals = _tf.map_fn(prob_fn_2, _tf.transpose(vals))

            n_data = _tf.shape(vals)[0]
            quant = self._spline.interpolate(
                prob_vals,
                _tf.transpose(self.warping.backward(vals)),
                _tf.tile(probabilities[:, None], [1, n_data])
            )

            # quant = _tf.map_fn(quant_fn, probabilities)
            quant = _tf.transpose(quant)

            # single point case
            # quant = _tf.cond(
            #     _tf.less(_tf.rank(quant), 2),
            #     lambda: _tf.expand_dims(quant, 0),
            #     lambda: quant)

            out["quantiles"] = quant

        return out


class HeteroscedasticGaussian(_ContinuousLikelihood):
    """
    Heteroscedastic Gaussian likelihood.

    Allows the modeling of spatially-dependent noise. Requires two latent
    variables, one for the mean and other for the log-variance. The analytical
    form uses approximations; better results are obtained by Monte Carlo.
    """
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        self._size = 2

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y_warped = self.warping.forward(y)
        y_derivative = self.warping.derivative(y)
        log_derivative = _tf.math.log(y_derivative)

        if self._use_monte_carlo:
            samples_mean = samples[:, 0, :]
            samples_noise = _tf.sqrt(_tf.exp(samples[:, 1, :]))

            distribution = self._make_distribution(samples_mean, samples_noise)

            log_density = distribution.log_prob(y_warped)
            log_density = _tf.math.reduce_mean(
                log_density, axis=1, keepdims=True)
        else:
            # double integral
            loc_mean = mu[:, 0]
            loc_var = var[:, 0]
            noise_mean = mu[:, 1]
            noise_var = var[:, 1]

            vals_1 = _tf.sqrt(2 * loc_var[:, None, None]) \
                     * _ROOTS_8[None, :, None] + loc_mean[:, None, None]
            vals_2 = _tf.sqrt(2 * noise_var[:, None, None]) \
                     * _ROOTS_8[None, None, :] + noise_mean[:, None, None]
            w = _WEIGHTS_8[None, :, None] * _WEIGHTS_8[None, None, :]
            w = w / _tf.reduce_sum(w)

            distribution = self._make_distribution(vals_1, vals_2)

            log_density = distribution.log_prob(y_warped[:, :, None])
            log_density = _tf.reduce_sum(log_density * w, axis=2)
            log_density = _tf.reduce_sum(log_density, axis=1,
                                         keepdims=True)

        lik = _tf.reduce_sum((log_density + log_derivative) * has_value)

        return lik

    def _make_distribution(self, loc, scale):
        return _tfd.Normal(loc, _tf.exp(scale/2))

    def predict(self, mu, var, sims, explained_var, *args, quantiles=None,
                probabilities=None, **kwargs):
        # double integral
        loc_mean = mu[:, 0]
        loc_var = var[:, 0]
        noise_mean = mu[:, 1]
        noise_var = var[:, 1]

        vals_1 = _tf.sqrt(2 * loc_var[:, None, None]) \
                 * _ROOTS_8[None, :, None] + loc_mean[:, None, None]
        vals_2 = _tf.sqrt(2 * noise_var[:, None, None]) \
                 * _ROOTS_8[None, None, :] + noise_mean[:, None, None]
        w = _WEIGHTS_8[None, :, None] * _WEIGHTS_8[None, None, :]
        w = _tf.reshape(w / _tf.reduce_sum(w), [-1])
        vals_1 = _tf.reshape(_tf.tile(vals_1, [1, 1, 8]), [-1, 8 * 8])
        vals_2 = _tf.reshape(_tf.tile(vals_2, [1, 8, 1]), [-1, 8 * 8])

        distribution = _tfd.MixtureSameFamily(
            _tfd.Categorical(probs=w),
            self._make_distribution(vals_1, vals_2)
        )

        lik_var = distribution.variance()
        weights = _tf.squeeze(explained_var[:, 0, None]) / (lik_var + 1e-6)

        out = {"mean": loc_mean,
               "variance": loc_var,
               "simulations": self.warping.backward(sims[:, 0, :]),
               "weights": _tf.squeeze(weights),
               }

        def prob_fn(q):
            q = _tf.expand_dims(q, 0)
            p = distribution.cdf(_tf.squeeze(self.warping.forward(q)))
            return p

        if quantiles is not None:
            prob = _tf.map_fn(prob_fn, quantiles)
            prob = _tf.transpose(prob)

            # single point case
            prob = _tf.cond(
                _tf.less(_tf.rank(prob), 2),
                lambda: _tf.expand_dims(prob, 0),
                lambda: prob)

            out["probabilities"] = prob

        if probabilities is not None:
            vals = _tf.constant(_np.linspace(-5, 5, 101)[None, :], _tf.float64)
            vals = _tf.sqrt(2 * loc_var[:, None]) * vals + loc_mean[:, None]
            warped_vals = self.warping.backward(vals)
            prob_vals = _tf.map_fn(lambda x: distribution.cdf(x),
                                   _tf.transpose(vals))

            n_data = _tf.shape(warped_vals)[0]
            quant = self._spline.interpolate(
                prob_vals,
                _tf.transpose(warped_vals),
                _tf.tile(probabilities[:, None], [1, n_data])
            )

            quant = _tf.transpose(quant)

            out["quantiles"] = quant

        return out


class HeteroscedasticLaplace(HeteroscedasticGaussian):
    def _make_distribution(self, loc, scale):
        return _tfd.Laplace(loc, _tf.exp(scale))


class Bernoulli(_Likelihood):
    def __init__(self, shift=0, sharpness=1):
        """
        Bernoulli likelihood.

        Used for binary categorical variables.

        Parameters
        ----------
        shift : double
            How much to favor the positive or negative class. Value between
            -5 and 5.
        sharpness : int
            Data augmentation. The weight of the data is multiplied by this
            factor. Results in sharper transitions between positive and
            negative regions.
        """
        super().__init__(1)
        self.sharpness = sharpness
        self._add_parameter("shift", _gpr.RealParameter(shift, -5, 5))
        self._add_parameter("slope", _gpr.PositiveParameter(1, 0.01, 100))

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        shift = self.parameters["shift"].get_value()
        slope = self.parameters["slope"].get_value()
        # distribution = _tfd.Normal(- shift, _tf.constant(1.0, _tf.float64))
        distribution = _tfd.Normal(- shift, 1 / slope)

        log_density = distribution.log_cdf(vals) * y \
                      + distribution.log_survival_function(vals) * (1 - y)
        log_density = _tf.reduce_sum(log_density * w, axis=1, keepdims=True)

        lik = _tf.reduce_sum(log_density * has_value)

        return lik * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        shift = self.parameters["shift"].get_value()
        slope = self.parameters["slope"].get_value()
        # distribution = _tfd.Normal(- shift, _tf.constant(1.0, _tf.float64))
        distribution = _tfd.Normal(- shift, 1 / slope)

        prob = distribution.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        entropy = (- prob * _tf.math.log(prob)
                   - (1 - prob) * _tf.math.log(1 - prob)) / _np.log(2)
        uncertainty = _tf.sqrt(_tf.squeeze(var) * entropy)

        prob_sims = distribution.cdf(sims)

        lik_var = prob * (1 - prob)
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        # weights = weights ** 2

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": prob_sims[:, 0, :],
               "probability": prob,
               "entropy": entropy,
               "uncertainty": uncertainty,
               "weights": _tf.squeeze(weights)}

        return out

    @classmethod
    def one_class(cls, sharpness=1):
        lik = Bernoulli(shift=-3, sharpness=sharpness)
        lik.parameters["shift"].fix()
        return lik


class BernoulliMaximumMargin(_Likelihood):
    def __init__(self):
        super().__init__(1)
        self._add_parameter("c_rate", _gpr.PositiveParameter(1, 1e-3, 1e3))

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        y = 2 * y - 1
        c_rate = self.parameters["c_rate"].get_value()

        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        log_density = _tf.where(
            _tf.less(_tf.math.abs(vals), 1.0),
            - _tf.math.log(1 + _tf.exp(-2 * c_rate * y * vals)),
            - _tf.math.log(1 + _tf.exp(- c_rate * y * (vals + _tf.sign(vals))))
        )
        log_density = _tf.reduce_sum(log_density * w, axis=1, keepdims=True)

        lik = _tf.reduce_sum(log_density * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        prob = self.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        prob_sims = self.cdf(sims)

        lik_var = prob * (1 - prob)
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        # weights = weights ** 2

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": prob_sims[:, 0, :],
               "probability": prob,
               "weights": _tf.squeeze(weights)}

        return out

    def cdf(self, x):
        c_rate = self.parameters["c_rate"].get_value()
        prob = _tf.where(
            _tf.less(_tf.math.abs(x), 1.0),
            1 / (1 + _tf.exp(-2 * c_rate * x)),
            1 / (1 + _tf.exp(- c_rate * (x + _tf.sign(x))))
        )
        return prob


class _CategoricalLikelihood(_Likelihood):
    def __init__(self, size, use_monte_carlo=False):
        super().__init__(size, use_monte_carlo)

    @staticmethod
    def entropy_and_indicators(probabilities, var, explained_var):
        n_cat = _tf.shape(probabilities)[1]
        log_n = _tf.math.log(_tf.cast(n_cat, _tf.float64))
        n_data = _tf.shape(probabilities)[0]

        entropy = - _tf.reduce_sum(
            probabilities * _tf.math.log(probabilities + 1e-6), axis=1) / log_n
        entropy = _tf.maximum(entropy, 0.0)
        avg_var = _tf.reduce_sum(var * probabilities, axis=1)
        uncertainty = _tf.sqrt(avg_var * entropy)
        indicators = _tf.math.log(probabilities + 1e-6)

        idx = _tf.range(n_data)[:, None]

        def ind_fn(z):
            idx_cat, col = z
            tmp_ind = _tf.tensor_scatter_nd_update(
                indicators,
                _tf.concat([idx, _tf.ones_like(idx) * idx_cat], axis=1),
                _tf.ones([n_data], _tf.float64) * -999
            )
            return col - _tf.reduce_max(tmp_ind, axis=1)

        ind_skew = _tf.map_fn(
            ind_fn, [_tf.range(n_cat), _tf.transpose(indicators)],
            dtype=_tf.float64)
        ind_skew = _tf.transpose(ind_skew)

        lik_var = probabilities * (1 - probabilities)
        lik_var = _tf.reduce_sum(lik_var, axis=1)
        weights = _tf.reduce_sum(explained_var, axis=1) / (lik_var + 1e-6)

        return entropy, uncertainty, ind_skew, weights


class CategoricalGaussianIndicator(_CategoricalLikelihood):
    def __init__(self, n_components, tol=1e-3, sharpness=1,
                 use_monte_carlo=False):
        """
        Gaussian likelihood for indicator variables.

        Assumes mutually exclusive categories (i.e. no geological rules),
        leading to maximum entropy far from the data points. Is capable of
        dealing with boundary data.

        Parameters
        ----------
        n_components : int
            The number of categories.
        tol : double
            Normal score tolerance for boundary data.
        sharpness : int
            Data augmentation. The weight of the data is multiplied by this
            factor. Results in sharper transitions between categories.
        """
        super().__init__(n_components, use_monte_carlo)
        self.tol = _tf.constant(tol, _tf.float64)
        self.sharpness = _tf.constant(sharpness, _tf.float64)

    def log_lik(self, mu, var, y, has_value, is_boundary=None,
                samples=None, *args, **kwargs):
        y = 2 * y - 1

        if self._use_monte_carlo:
            # pos = _tf.where(
            #     _tf.greater(samples, self.tol),
            #     _tf.ones_like(samples),
            #     _tf.zeros_like(samples)
            # )
            # neg = _tf.where(
            #     _tf.less(samples, - self.tol),
            #     _tf.ones_like(samples),
            #     _tf.zeros_like(samples)
            # )
            #
            # prob_pos = _tf.reduce_mean(pos, axis=-1)
            # prob_neg = _tf.reduce_mean(neg, axis=-1)
            # prob_zero = 1 - prob_neg - prob_pos
            #
            # prob_pos = _tf.math.log(prob_pos + 1e-6)
            # prob_neg = _tf.math.log(prob_neg + 1e-6)
            # prob_zero = _tf.math.log(prob_zero + 1e-6)

            dist = _tfd.Normal(samples, 1.0)  #self.tol)
            prob_neg = dist.log_cdf(- self.tol)
            prob_zero = _tf.math.log(
                dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
            prob_pos = dist.log_survival_function(self.tol)

            # prob_neg = _tf.reduce_mean(prob_neg, axis=-1)
            # prob_pos = _tf.reduce_mean(prob_pos, axis=-1)
            # prob_zero = _tf.reduce_mean(prob_zero, axis=-1)

            n_sim = _tf.shape(samples)[2]
            y = _tf.tile(y[:, :, None], [1, 1, n_sim])

            log_density = _tf.where(
                _tf.less(y, - self.tol),
                prob_neg,
                _tf.where(_tf.greater(y, self.tol),
                          prob_pos,
                          prob_zero)
            )
            log_density = _tf.reduce_mean(log_density, axis=2)

        else:
            dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))
            prob_neg = dist.log_cdf(- self.tol)
            prob_zero = _tf.math.log(
                dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
            prob_pos = dist.log_survival_function(self.tol)

            log_density = _tf.where(
                _tf.less(y, - self.tol),
                prob_neg,
                _tf.where(_tf.greater(y, self.tol),
                          prob_pos,
                          prob_zero)
            )

            # log_density_2 = _tf.math.log(- _tf.math.expm1(log_density))
            # log_density = log_density - log_density_2 * 1e-4

        log_density = _tf.reduce_sum(log_density, axis=1, keepdims=True)

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)
        log_density = _tf.reduce_sum(log_density * has_value)

        return log_density * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        # n_cat = _tf.shape(mu)[1]
        # n_data = _tf.shape(mu)[0]

        dist = _tfd.Normal(mu, _tf.sqrt(var))
        log_prob_positive = dist.log_survival_function(self.tol)
        log_prob_negative = dist.log_prob(- self.tol)

        # probability of being class i AND not being the others
        log_prob_final = []
        for i in range(self.size):
            log_prob_i = []
            for j in range(self.size):
                if j == i:
                    log_prob_i.append(log_prob_positive[:, j])
                else:
                    log_prob_i.append(log_prob_negative[:, j])
            log_prob_final.append(_tf.add_n(log_prob_i))
        log_prob_final = _tf.stack(log_prob_final, axis=1)

        # prob = _tf.nn.softmax(log_prob_positive, axis=1)
        prob = _tf.nn.softmax(log_prob_final, axis=1)

        entropy, uncertainty, ind_skew, weights = self.entropy_and_indicators(
            prob, var, explained_var
        )

        output = {"mean": mu,
                  "variance": var,
                  "probability": prob,
                  "simulations": sims,
                  "entropy": entropy,
                  "uncertainty": uncertainty,
                  "indicators": ind_skew,
                  "weights": weights}
        return output


class SequentialGaussianIndicator(CategoricalGaussianIndicator):
    # def __init__(self, n_components, tol=1e-3, sharpness=1):
    #     super().__init__(n_components - 1, tol, sharpness)

    def log_lik(self, mu, var, y, has_value, is_boundary=None,
                samples=None, *args, **kwargs):
        y = 2 * y - 1

        n_data = _tf.shape(mu)[0]

        # sequential logic
        # ones = _tf.ones_like(y[:, 0])
        # zeros = _tf.zeros_like(y[:, 0])
        # keep_vals = [_tf.where(_tf.greater(y[:, 0], 0.0), ones, zeros)]
        # contacts = _tf.where(_tf.equal(y[:, 0], 0.0), ones, zeros)
        # for i in range(1, self.size):
        #     # positives for class i
        #     k = _tf.where(_tf.greater(y[:, i], 0.0), ones, zeros)
        #
        #     # negatives up to class i-1
        #     k = _tf.where(_tf.logical_or(
        #         _tf.equal(k, 1.0), _tf.equal(keep_vals[i - 1], 1.0)),
        #         ones, zeros
        #     )
        #
        #     # contacts up to class i-1
        #     contacts = _tf.where(
        #         _tf.logical_or(
        #             _tf.equal(contacts, 1.0),
        #             _tf.equal(y[:, i], 0.0)
        #         ),
        #         ones, zeros
        #     )
        #     k = _tf.where(_tf.logical_or(
        #         _tf.equal(k, 1.0), _tf.equal(contacts, 1.0)),
        #         ones, zeros
        #     )
        #     keep_vals.append(k)
        # keep_vals = _tf.stack(keep_vals, axis=1)

        # mu = _tf.concat([
        #     _tf.ones([n_data, 1], _tf.float64),
        #     mu
        # ], axis=1)
        # var = _tf.concat([
        #     _tf.ones([n_data, 1], _tf.float64) * 1e-6,
        #     var
        # ], axis=1)

        # dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))
        # prob_pos = dist.survival_function(self.tol)

        # # interference over previous classes
        # mu_int = [mu[:, 0, None]]
        # var_int = [var[:, 0, None]]
        # for i in range(1, self.size):
        #     dist_int = _tfd.Normal(
        #         _tf.concat(mu_int, axis=1),
        #         _tf.sqrt(_tf.concat(var_int, axis=1) + 1e-6))
        #     prob_pos_int = dist_int.survival_function(self.tol)
        #     for j in range(i):
        #         w = 1.0 - 2*prob_pos[:, i, None]*prob_pos_int[:, j, None]
        #         mu_int[j] = mu_int[j] * w
        #         var_int[j] = var_int[j] * w**2
        #     mu_int.append(mu[:, i, None])
        #     var_int.append(var[:, i, None])
        # mu_int = _tf.concat(mu_int, axis=1)
        # var_int = _tf.concat(var_int, axis=1)
        #
        # dist = _tfd.Normal(mu_int, _tf.sqrt(var_int + 1e-6))
        # prob_neg = dist.log_cdf(- self.tol)
        # prob_zero = _tf.math.log(
        #     dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
        # prob_pos = dist.log_survival_function(self.tol)

        dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))
        prob_neg = dist.cdf(- self.tol)
        prob_pos = dist.survival_function(self.tol)

        prob_neg = _tf.unstack(prob_neg, axis=1)
        prob_pos = _tf.unstack(prob_pos, axis=1)
        for i in range(1, self.size):
            for j in range(i):
                prob_neg[j] = prob_neg[j] * prob_neg[i] + prob_pos[i]
                prob_pos[j] = prob_pos[j] * prob_neg[i]
        prob_neg = _tf.stack(prob_neg, axis=1)
        prob_pos = _tf.stack(prob_pos, axis=1)

        prob_zero = _tf.math.log(1 - prob_pos - prob_neg + 1e-6)
        prob_neg = _tf.math.log(prob_neg + 1e-6)
        prob_pos = _tf.math.log(prob_pos + 1e-6)

        log_density = _tf.where(
            _tf.less(y, - self.tol),
            prob_neg,
            _tf.where(_tf.greater(y, self.tol),
                      prob_pos,
                      prob_zero)
        )

        log_density = _tf.reduce_sum(log_density, axis=1, keepdims=True)

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)
        log_density = _tf.reduce_sum(log_density * has_value)

        return log_density * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        # n_data = _tf.shape(mu)[0]
        # mu = _tf.concat([
        #     _tf.ones([n_data, 1], _tf.float64),
        #     mu
        # ], axis=1)
        # var = _tf.concat([
        #     _tf.ones([n_data, 1], _tf.float64) * 1e-6,
        #     var
        # ], axis=1)

        # dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))
        # prob_pos = dist.survival_function(self.tol)
        #
        # # interference over previous classes
        # mu_int = [mu[:, 0, None]]
        # var_int = [var[:, 0, None]]
        # for i in range(1, self.size):
        #     dist_int = _tfd.Normal(
        #         _tf.concat(mu_int, axis=1),
        #         _tf.sqrt(_tf.concat(var_int, axis=1) + 1e-6))
        #     prob_pos_int = dist_int.survival_function(self.tol)
        #     for j in range(i):
        #         w = 1.0 - 2 * prob_pos[:, i, None] * prob_pos_int[:, j, None]
        #         mu_int[j] = mu_int[j] * w
        #         var_int[j] = var_int[j] * w ** 2
        #     mu_int.append(mu[:, i, None])
        #     var_int.append(var[:, i, None])
        # mu_int = _tf.concat(mu_int, axis=1)
        # var_int = _tf.concat(var_int, axis=1)
        #
        # dist = _tfd.Normal(mu_int, _tf.sqrt(var_int + 1e-6))
        # log_prob_positive = dist.log_survival_function(self.tol)
        # log_prob_negative = dist.log_prob(- self.tol)

        dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))
        prob_neg = dist.cdf(0)
        prob_pos = dist.survival_function(0)

        prob_neg = _tf.unstack(prob_neg, axis=1)
        prob_pos = _tf.unstack(prob_pos, axis=1)
        for i in range(1, self.size):
            for j in range(i):
                prob_neg[j] = prob_neg[j] * prob_neg[i] + prob_pos[i]
                prob_pos[j] = prob_pos[j] * prob_neg[i]
        prob_neg = _tf.stack(prob_neg, axis=1)
        prob_pos = _tf.stack(prob_pos, axis=1)
        log_prob_negative = _tf.math.log(prob_neg + 1e-6)
        log_prob_positive = _tf.math.log(prob_pos + 1e-6)

        # probability of being class i AND not being the others
        log_prob_final = []
        for i in range(self.size):
            log_prob_i = []
            for j in range(self.size):
                if j == i:
                    log_prob_i.append(log_prob_positive[:, j])
                else:
                    log_prob_i.append(log_prob_negative[:, j])
            log_prob_final.append(_tf.add_n(log_prob_i))
        log_prob_final = _tf.stack(log_prob_final, axis=1)

        prob = _tf.nn.softmax(log_prob_final, axis=1)

        entropy, uncertainty, ind_skew, weights = self.entropy_and_indicators(
            prob, var, explained_var
        )

        output = {"mean": mu,
                  "variance": var,
                  "probability": prob,
                  "simulations": sims,
                  "entropy": entropy,
                  "uncertainty": uncertainty,
                  "indicators": ind_skew,
                  "weights": weights}
        return output


class _CompositionalLikelihood(_Likelihood):
    def __init__(self, n_components, n_basis=None, contrast_matrix=None,
                 warping=_warp.Identity()):
        """
        Compositional likelihood.

        Used to model compositional variables (i.e. multiple variables which
        are constrained to unit sum). Assumes the data sums to 1. Requires up
        to `N-1` latent variables for `N` components. Due to the non-linear
        constraint, employs Monte Carlo for training.

        A good contrast matrix can improve the modeling results. See the
        compositional data literature for how to build one. Good contrasts
        often involve major vs minor elements, oxides vs sulfides, etc.

        Parameters
        ----------
        n_components : int
            Number of components in the composition.
        n_basis : int
            Number of latent variables to use. Default is `n_components-1`. A
            smaller value forces a PCA-like decomposition of the contrasts.
        contrast_matrix : array
            A `n_components-1` by `n_components` matrix of contrasts.
        warping : Warping
            A warping object to be applied to each contrast, trained
            independently.
        """
        if n_basis is None:
            n_basis = n_components - 1
        elif n_basis >= n_components:
            raise ValueError("n_basis must be smaller than n_components")
        self.n_basis = n_basis

        super().__init__(n_basis)
        self.n_components = n_components

        self._add_parameter(
            "basis",
            _gpr.OrthonormalMatrix(n_components - 1, n_basis)
        )

        if contrast_matrix is None:
            contrast_matrix = _helmert(n_components)
        elif contrast_matrix.shape != (n_components - 1, n_components):
            raise ValueError("invalid shape for contrast_matrix; "
                             "shape must be (n_components - 1, n_components)")
        self.contrast = _tf.constant(contrast_matrix, _tf.float64)

        self.warpings = [self._register(_copy.deepcopy(warping))
                         for _ in range(n_basis)]

    def _make_distribution(self, *args, **kwargs):
        raise NotImplementedError

    def white_noise(self, shape, seed, coherent_noise=False, **kwargs):
        n_data = shape[0] if not coherent_noise else 1
        n_var = shape[1]
        n_samples = shape[2]

        dist = self._make_distribution(_tf.constant(0.0, dtype=_tf.float64))

        if coherent_noise:
            rnd = _tf.random.stateless_uniform([n_data, n_var, n_samples], seed=[seed, 0], dtype=_tf.float64)
        else:
            rnd = _tf.random.uniform([n_data, n_var, n_samples], seed=seed, dtype=_tf.float64)

        sample = dist.quantile(rnd)
        return sample

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        basis = self.parameters["basis"].get_value()

        y_ilr = _tf.matmul(_tf.math.log(y), self.contrast, False, True)
        y_ilr = _tf.matmul(y_ilr, basis)
        y_split = _tf.split(y_ilr, self.size, axis=1)

        y_warped = _tf.concat(
            [wp.forward(z) for wp, z in zip(self.warpings, y_split)],
            axis=1)
        y_derivative = [wp.derivative(z)
                        for wp, z in zip(self.warpings, y_split)]
        log_derivative = _tf.math.log(_tf.concat(y_derivative, axis=1))

        distribution = self._make_distribution(samples)

        log_density = distribution.log_prob(y_warped[:, :, None])
        log_density = _tf.math.reduce_mean(
            log_density, axis=2, keepdims=False)

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)

        lik = _tf.reduce_sum((log_density + log_derivative) * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var,
                *args, quantiles=None, include_noise=True,
                **kwargs):
        basis = self.parameters["basis"].get_value()
        rotated_contrast = _tf.matmul(basis, self.contrast, True)

        # ignores warping, used only for calculating weights
        mu = _tf.matmul(mu, rotated_contrast)
        var = _tf.matmul(var, rotated_contrast ** 2)
        explained_var = _tf.matmul(explained_var, rotated_contrast ** 2)

        if include_noise:
            coherent_sims = sims + self.white_noise(_tf.shape(sims), seed=1234, coherent_noise=True)
            rough_sims = sims + self.white_noise(_tf.shape(sims), seed=1234, coherent_noise=False)
        else:
            coherent_sims = sims
            rough_sims = sims

        def reverse_comp(x):
            # backward warping
            x = _tf.split(x, self.size, axis=1)
            x = [wp.backward(s[:, 0, :]) for wp, s in zip(self.warpings, x)]
            x = _tf.stack(x, axis=1)

            # reverse ilr
            sims_clr = _tf.einsum("nij,ik->nkj", x, rotated_contrast)
            sims_simplex = _tf.nn.softmax(sims_clr, axis=1)

            return sims_simplex

        coherent_sims = reverse_comp(coherent_sims)
        rough_sims = reverse_comp(rough_sims)

        # average composition (euclidean)
        # avg = _tf.reduce_mean(sims_clr, axis=2)
        # prob = _tf.nn.softmax(avg, axis=1)

        # average composition (simplex)
        prob = _tf.reduce_mean(coherent_sims, axis=2)

        weights = _tf.reduce_sum(explained_var, axis=1) \
                  / (_tf.reduce_sum(var, axis=1) + 1e-6)

        out = {"mean": mu,
               "variance": var,
               "simulations": rough_sims,
               "probability": prob,
               "weights": weights}

        return out

    def initialize(self, y):
        y_ilr = _tf.matmul(_tf.math.log(y), self.contrast, False, True)

        y_centered = y_ilr - _tf.reduce_mean(y_ilr, axis=0, keepdims=True)
        cov_mat = _tf.matmul(y_centered, y_centered, True)
        vals, vecs = _tf.linalg.eigh(cov_mat)
        vecs = vecs[:, ::-1][:, :self.size]

        self.parameters["basis"].set_value(vecs.numpy())
        self.parameters["basis"].fix()

        y_ilr = _tf.matmul(y_ilr, vecs)

        for i, wp in enumerate(self.warpings):
            wp.initialize(y_ilr[:, i])


class CompositionalGaussian(_CompositionalLikelihood):
    def __init__(self, n_components, n_basis=None, contrast_matrix=None,
                 tie_components=True, warping=_warp.Center()):
        super().__init__(n_components, n_basis, contrast_matrix, warping)
        if tie_components:
            self._add_parameter(
                "noise",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, 1]) * 0.1,
                    _np.ones([1, 1, 1]) * 1e-6,
                    _np.ones([1, 1, 1]) * 10))
        else:
            self._add_parameter(
                "noise",
                _gpr.PositiveParameter(
                    _np.ones([1, self.size, 1]) * 0.1,
                    _np.ones([1, self.size, 1]) * 1e-6,
                    _np.ones([1, self.size, 1]) * 10))

    def _make_distribution(self, loc):
        return _tfd.Normal(loc, _tf.sqrt(self.parameters["noise"].get_value()[0, :, :]))


class CompositionalLaplace(_CompositionalLikelihood):
    def __init__(self, n_components, n_basis=None, contrast_matrix=None,
                 tie_components=True, warping=_warp.Center()):
        super().__init__(n_components, n_basis, contrast_matrix, warping)
        if tie_components:
            self._add_parameter(
                "rate",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, 1]) * 0.1,
                    _np.ones([1, 1, 1]) * 1e-6,
                    _np.ones([1, 1, 1]) * 10))
        else:
            self._add_parameter(
                "rate",
                _gpr.PositiveParameter(
                    _np.ones([1, self.size, 1]) * 0.1,
                    _np.ones([1, self.size, 1]) * 1e-6,
                    _np.ones([1, self.size, 1]) * 10))

    def _make_distribution(self, loc):
        return _tfd.Laplace(loc, self.parameters["rate"].get_value()[0, :, :])


class CompositionalEpsilonInsensitive(_CompositionalLikelihood):
    def __init__(self, n_components, n_basis=None, contrast_matrix=None,
                 tie_components=True, warping=_warp.Center()):
        super().__init__(n_components, n_basis, contrast_matrix, warping)
        if tie_components:
            self._add_parameter(
                "epsilon",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, 1]) * 1e-3,
                    _np.ones([1, 1, 1]) * 1e-6,
                    _np.ones([1, 1, 1]) * 10))
            self._add_parameter(
                "c_rate",
                _gpr.PositiveParameter(
                    _np.ones([1, 1, 1]),
                    _np.ones([1, 1, 1]) * 1e-3,
                    _np.ones([1, 1, 1]) * 1e3))
        else:
            self._add_parameter(
                "epsilon",
                _gpr.PositiveParameter(
                    _np.ones([1, self.size, 1]) * 1e-3,
                    _np.ones([1, self.size, 1]) * 1e-6,
                    _np.ones([1, self.size, 1]) * 10))
            self._add_parameter(
                "c_rate",
                _gpr.PositiveParameter(
                    _np.ones([1, self.size, 1]),
                    _np.ones([1, self.size, 1]) * 1e-3,
                    _np.ones([1, self.size, 1]) * 1e3))

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        basis = self.parameters["basis"].get_value()

        y_ilr = _tf.matmul(_tf.math.log(y), self.contrast, False, True)
        y_ilr = _tf.matmul(y_ilr, basis)
        y_split = _tf.split(y_ilr, self.size, axis=1)

        y_warped = _tf.concat(
            [wp.forward(z) for wp, z in zip(self.warpings, y_split)],
            axis=1)
        y_derivative = [wp.derivative(z)
                        for wp, z in zip(self.warpings, y_split)]
        log_derivative = _tf.math.log(_tf.concat(y_derivative, axis=1))

        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        y_centered = _tf.math.abs(y_warped[:, :, None] - samples)

        log_density = _tf.where(
            _tf.less_equal(y_centered, epsilon),
            _tf.zeros_like(y_centered),
            - c_rate * (y_centered - epsilon)
        )
        log_density = log_density - _tf.math.log(2 * (epsilon + 1 / c_rate))
        log_density = _tf.math.reduce_mean(
            log_density, axis=2, keepdims=False)

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)

        lik = _tf.reduce_sum((log_density + log_derivative) * has_value)

        return lik

    def inv_cdf(self, p):
        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        area = 2 * (epsilon + 1 / c_rate)
        # p_1 = (1/c_rate - 2*epsilon) / area
        # p_2 = (1/c_rate) / area
        p_1 = (1 / c_rate) / area
        p_2 = p_1 + 2 * epsilon / area

        val_1 = _tf.math.log(area * c_rate * p) / c_rate - epsilon
        val_2 = area * p - 1 / c_rate - epsilon
        val_3 = epsilon - _tf.math.log(2 - c_rate * (area * p - 2 * epsilon)) / c_rate

        x = _tf.where(_tf.greater(p, p_1), val_2, val_1)
        x = _tf.where(_tf.greater(p, p_2), val_3, x)
        return x

    def white_noise(self, shape, seed, coherent_noise=False, **kwargs):
        if coherent_noise:
            rnd = _tf.random.stateless_uniform([1, shape[2]], minval=0.0, maxval=1.0, seed=[seed, 0], dtype=_tf.float64)
        else:
            rnd = _tf.random.uniform(shape, minval=0.0, maxval=1.0, seed=seed, dtype=_tf.float64)

        sample = self.inv_cdf(rnd)
        return sample


class OrderedGaussianIndicator(_CategoricalLikelihood):
    def __init__(self, levels, tol=1e-6, sharpness=1):
        super().__init__(1)
        self.levels = levels
        self.tol = tol
        self.sharpness = sharpness

        if levels > 1:
            self._add_parameter(
                "thresholds",
                _gpr.CompositionalParameter(
                    _np.ones([levels - 1]) / (levels - 1))
            )

    def get_thresholds(self):
        thresholds = self.parameters["thresholds"].get_value()
        thresholds = _tf.concat([
            _tf.constant([0.0], _tf.float64),
            _tf.cumsum(thresholds) * (self.levels - 1)
        ], axis=0)
        return thresholds

    def log_lik(self, mu, var, y, has_value,
                samples=None, *args, **kwargs):
        mu = mu + (self.levels - 1) / 2
        var = var * self.levels**2

        dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))

        log_density = _tf.zeros_like(mu)

        if self.levels == 1:
            prob_zero = _tf.math.log(
                dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
            prob_neg = dist.log_cdf(- self.tol)
            prob_pos = dist.log_survival_function(self.tol)

            log_density = _tf.where(
                _tf.less(y, - self.tol),
                prob_neg,
                _tf.where(
                    _tf.greater(y, self.tol),
                    prob_pos,
                    prob_zero
                )
            )

        else:
            thresholds = self.get_thresholds()

            for i in range(self.levels):
                prob_zero = _tf.math.log(
                    dist.cdf(thresholds[i] + self.tol)
                    - dist.cdf(thresholds[i] - self.tol) + 1e-6)

                if i == 0:
                    prob_neg = dist.log_cdf(- self.tol)
                    prob_pos = _tf.math.log(
                        dist.survival_function(thresholds[i] + self.tol)
                        - dist.survival_function(thresholds[i + 1] - self.tol)
                        + 1e-6
                    )
                elif i == self.levels - 1:
                    prob_neg = _tf.math.log(
                        dist.cdf(thresholds[i] - self.tol)
                        - dist.cdf(thresholds[i - 1] + self.tol)
                        + 1e-6
                    )
                    prob_pos = dist.log_survival_function(
                        thresholds[i] + self.tol)
                else:
                    prob_neg = _tf.math.log(
                        dist.cdf(thresholds[i] - self.tol)
                        - dist.cdf(thresholds[i - 1] + self.tol)
                        + 1e-6
                    )
                    prob_pos = _tf.math.log(
                        dist.survival_function(thresholds[i] + self.tol)
                        - dist.survival_function(thresholds[i + 1] - self.tol)
                        + 1e-6
                    )

                log_density = _tf.where(
                    _tf.logical_and(
                        _tf.less(y, i - self.tol),
                        _tf.greater(y, i - 1 + self.tol)
                    ),
                    prob_neg,
                    _tf.where(
                        _tf.logical_and(
                            _tf.greater(y, i + self.tol),
                            _tf.less(y, i + 1 - self.tol)
                        ),
                        prob_pos,
                        _tf.where(
                            _tf.logical_and(
                                _tf.greater(y, i - self.tol),
                                _tf.less(y, i + self.tol)
                            ),
                            prob_zero,
                            log_density
                        )
                    )
                )

        # log_density_2 = _tf.math.log(- _tf.math.expm1(log_density))
        # log_density = log_density - log_density_2 #* 1e-2

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)

        # weights = 2 - _tf.math.exp(log_density)
        # weights = weights / _tf.reduce_sum(weights * has_value) \
        #           * _tf.reduce_sum(has_value)

        log_density = _tf.reduce_sum(log_density * has_value) # * weights)

        return log_density * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        mu = mu + (self.levels - 1) / 2
        # var = var * self.levels ** 2
        # explained_var = explained_var * self.levels ** 2
        sims = sims + (self.levels - 1) / 2

        dist = _tfd.Normal(mu, _tf.sqrt(var * self.levels ** 2 + 1e-6))

        if self.levels == 1:
            prob = [dist.cdf(0), dist.survival_function(0)]
        else:
            thresholds = self.get_thresholds()

            prob = [dist.cdf(0)]
            for i in range(self.levels - 1):
                prob.append(dist.cdf(thresholds[i + 1])
                            - dist.cdf(thresholds[i]))
            prob.append(dist.survival_function(self.levels - 1))
        prob = _tf.concat(prob, axis=1)

        entropy, uncertainty, ind_skew, weights = self.entropy_and_indicators(
            prob, var, explained_var
        )

        # if self.levels > 1:
        #     mu = mu - thresholds[None, :]
        #     sims = sims - thresholds[None, :, None]

        output = {"mean": _tf.tile(mu, [1, self.levels + 1]),
                  "variance": _tf.tile(var, [1, self.levels + 1]),
                  "simulations": _tf.tile(sims, [1, self.levels + 1, 1]),
                  "weights": weights,
                  "probability": prob,
                  "entropy": entropy,
                  "uncertainty": uncertainty,
                  "indicators": ind_skew}
        return output


class GradientIndicator(_Likelihood):
    def __init__(self, tol=1e-3):
        super().__init__(1)
        self.tol = tol

    def log_lik(self, mu, var, y, has_value,
                samples=None, *args, **kwargs):

        dist = _tfd.Normal(mu, _tf.sqrt(var + 1e-6))

        prob_zero = _tf.math.log(
            dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
        prob_neg = dist.log_cdf(- self.tol)
        prob_pos = dist.log_survival_function(self.tol)

        log_density = _tf.where(
            _tf.less(y, - self.tol),
            prob_neg,
            _tf.where(
                _tf.greater(y, self.tol),
                prob_pos,
                prob_zero
            )
        )
        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)
        log_density = _tf.reduce_sum(log_density * has_value)

        return log_density

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        weights = _tf.squeeze(explained_var / (var + 1e-6))

        output = {"mean": _tf.squeeze(mu),
                  "variance": _tf.squeeze(var),
                  "simulations": sims[:, 0, :],
                  "weights": weights}
        return output
