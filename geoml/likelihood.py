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
# MERCHANTABILITY or FITNESS FOR matrix PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import geoml.warping as _warp
import geoml.parameter as _gpr
# import geoml.tftools as _tftools
import geoml.interpolation as _gint

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp
_tfd = _tfp.distributions


_ROOTS = _tf.constant(dtype=_tf.float64, value=[
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
_ROOTS = _tf.concat([-_ROOTS[::-1], _ROOTS], axis=0)

_WEIGHTS = _tf.constant(dtype=_tf.float64, value=[
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
_WEIGHTS = _tf.concat([_WEIGHTS[::-1], _WEIGHTS], axis=0)
_WEIGHTS = _WEIGHTS / _tf.reduce_sum(_WEIGHTS)


class _Likelihood:
    def __init__(self):
        self._all_parameters = []
        self.parameters = {}

    @property
    def all_parameters(self):
        return self._all_parameters

    def get_parameter_values(self, complete=False):
        value = []
        shape = []
        position = []
        min_val = []
        max_val = []

        for index, parameter in enumerate(self._all_parameters):
            if (not parameter.fixed) | complete:
                value.append(_tf.reshape(parameter.variable, [-1]).
                             numpy())
                shape.append(_tf.shape(parameter.variable).numpy())
                position.append(index)
                min_val.append(_tf.reshape(parameter.min_transformed, [-1]).
                               numpy())
                max_val.append(_tf.reshape(parameter.max_transformed, [-1]).
                               numpy())

        min_val = _np.concatenate(min_val, axis=0)
        max_val = _np.concatenate(max_val, axis=0)
        value = _np.concatenate(value, axis=0)

        return value, shape, position, min_val, max_val

    def update_parameters(self, value, shape, position):
        sizes = _np.array([int(_np.prod(sh)) for sh in shape])
        value = _np.split(value, _np.cumsum(sizes))[:-1]
        value = [_np.squeeze(val) if len(sh) == 0 else val
                 for val, sh in zip(value, shape)]

        for val, sh, pos in zip(value, shape, position):
            self._all_parameters[pos].set_value(
                _np.reshape(val, sh) if len(sh) > 0 else val,
                transformed=True
            )

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        raise NotImplementedError

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        raise NotImplementedError

    def log_lik_from_samples(self, samples, y, has_value, *args, **kwargs):
        raise NotImplementedError

    def predict_from_samples(self, samples):
        raise NotImplementedError


class _ContinuousLikelihood(_Likelihood):
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__()
        self._all_parameters += warping.all_parameters
        self.warping = warping
        self._use_monte_carlo = use_monte_carlo
        self._spline = _gint.MonotonicCubicSpline()

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
            vals = _tf.expand_dims(_ROOTS, axis=0)
            vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
            w = _tf.expand_dims(_WEIGHTS, axis=0)

            distribution = self._make_distribution(vals)

            log_density = distribution.log_prob(y_warped)
            log_density = _tf.reduce_sum(log_density * w, axis=1, keepdims=True)

        lik = _tf.reduce_sum((log_density + log_derivative) * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var, *args, quantiles=None,
                probabilities=None, **kwargs):
        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        # w = _tf.expand_dims(_WEIGHTS, axis=0)

        # distribution = self._make_distribution(vals)
        distribution = _tfd.MixtureSameFamily(
            _tfd.Categorical(probs=_WEIGHTS),
            self._make_distribution(vals)
        )

        lik_var = distribution.variance()
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        weights = weights**2

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
            # prob_vals = distribution.cdf(vals)
            prob_vals = _tf.map_fn(lambda x: distribution.cdf(x),
                                   _tf.transpose(vals))

            n_data = _tf.shape(vals)[0]
            quant = self._spline.interpolate(
                prob_vals,
                _tf.transpose(vals),
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
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        noise = _gpr.PositiveParameter(0.1, 1e-12, 10)
        self.parameters.update({"noise": noise})
        self._all_parameters.append(noise)

    def _make_distribution(self, loc):
        return _tfd.Normal(loc, _tf.sqrt(self.parameters["noise"].get_value()))


class Laplace(_ContinuousLikelihood):
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        scale = _gpr.PositiveParameter(0.1, 1e-9, 10)
        self.parameters.update({"scale": scale})
        self._all_parameters.append(scale)

    def _make_distribution(self, loc):
        return _tfd.Laplace(loc, self.parameters["scale"].get_value())


class Gamma(_ContinuousLikelihood):
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        mean_alpha = _gpr.RealParameter(0, -3, 3)
        self.parameters.update({"mean_alpha": mean_alpha})
        self._all_parameters.append(mean_alpha)

    def _make_distribution(self, loc):
        mean_alpha = self.parameters["mean_alpha"].get_value()
        return _tfd.Gamma(_tf.exp(loc + mean_alpha) + 0.01,
                          _tf.constant(1.0, _tf.float64))


class Beta(_ContinuousLikelihood):
    def __init__(self, concentration=1, use_monte_carlo=False):
        super().__init__(use_monte_carlo=use_monte_carlo)
        concentration = _gpr.PositiveParameter(
            concentration, 1e-3, 10, fixed=False)
        self.parameters.update({"concentration": concentration})
        self._all_parameters.append(concentration)

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
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        scale = _gpr.PositiveParameter(0.1, 1e-9, 10)
        df = _gpr.PositiveParameter(5.0, 2.01, 50.0)
        self.parameters.update({"scale": scale, "df": df})
        self._all_parameters.append(scale)
        self._all_parameters.append(df)

    def _make_distribution(self, loc):
        return _tfd.StudentT(
            df=self.parameters["df"].get_value(),
            loc=loc,
            scale=self.parameters["scale"].get_value())


class EpsilonInsensitive(_ContinuousLikelihood):
    def __init__(self, warping=_warp.Identity(), use_monte_carlo=False):
        super().__init__(warping, use_monte_carlo)
        epsilon = _gpr.PositiveParameter(0.001, 1e-9, 10)
        c_rate = _gpr.PositiveParameter(1, 1e-3, 1e3, fixed=False)
        self.parameters.update({"epsilon": epsilon, "c_rate": c_rate})
        self._all_parameters.append(epsilon)
        self._all_parameters.append(c_rate)

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
            log_density = log_density - _tf.math.log(2 * (epsilon + 1 / c_rate))
            log_density = _tf.reduce_mean(log_density, axis=1, keepdims=True)

        else:
            vals = _tf.expand_dims(_ROOTS, axis=0)
            vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
            w = _tf.expand_dims(_WEIGHTS, axis=0)

            y_centered = _tf.math.abs(y_warped - vals)

            log_density = _tf.where(
                _tf.less_equal(y_centered, epsilon),
                _tf.zeros_like(y_centered),
                - c_rate * (y_centered - epsilon)
            )
            log_density = log_density - _tf.math.log(2 * (epsilon + 1 / c_rate))
            log_density = _tf.math.reduce_logsumexp(
                log_density + _tf.math.log(w), axis=1, keepdims=True)

        lik = _tf.reduce_sum(
            (log_density + log_derivative) * has_value)

        return lik

    def cdf(self, x):
        epsilon = self.parameters["epsilon"].get_value()
        c_rate = self.parameters["c_rate"].get_value()

        val_1 = _tf.math.exp(c_rate * (x - epsilon)) / c_rate
        val_2 = 1 / c_rate + x - epsilon
        val_3 = 2 * epsilon + 2 / c_rate \
                * (1 - 0.5 * _tf.math.exp(- c_rate * (x - epsilon)))

        prob = _tf.where(_tf.greater(x, -epsilon), val_2, val_1)
        prob = _tf.where(_tf.greater(x, epsilon), val_3, prob)

        prob = prob * 0.5 / (epsilon + 1 / c_rate)
        return prob

    def variance(self):
        e = self.parameters["epsilon"].get_value()
        c = self.parameters["c_rate"].get_value()

        n = 3 * c * e * (c * e + 2) + 6 + (c * e) ** 3
        d = 3 * c ** 2 * (c * e + 1)
        return n / d

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

        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS, axis=0)

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
                _tf.transpose(vals),
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


class Bernoulli(_Likelihood):
    def __init__(self, shift=0):
        super().__init__()
        shift = _gpr.RealParameter(shift, -5, 5)
        slope = _gpr.PositiveParameter(1, 0.01, 100)
        self.parameters.update({"shift": shift, "slope": slope})
        self._all_parameters.append(shift)
        self._all_parameters.append(slope)

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS, axis=0)

        shift = self.parameters["shift"].get_value()
        slope = self.parameters["slope"].get_value()
        # distribution = _tfd.Normal(- shift, _tf.constant(1.0, _tf.float64))
        distribution = _tfd.Normal(- shift, 1 / slope)

        log_density = distribution.log_cdf(vals) * y \
                      + distribution.log_survival_function(vals) * (1 - y)
        log_density = _tf.reduce_sum(log_density * w, axis=1, keepdims=True)

        lik = _tf.reduce_sum(log_density * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS, axis=0)

        shift = self.parameters["shift"].get_value()
        slope = self.parameters["slope"].get_value()
        # distribution = _tfd.Normal(- shift, _tf.constant(1.0, _tf.float64))
        distribution = _tfd.Normal(- shift, 1 / slope)

        prob = distribution.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        prob_sims = distribution.cdf(sims)

        lik_var = prob * (1 - prob)
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        weights = weights ** 2

        out = {"mean": _tf.squeeze(mu),
               "variance": _tf.squeeze(var),
               "simulations": prob_sims[:, 0, :],
               "probability": prob,
               "weights": _tf.squeeze(weights)}

        return out

    @classmethod
    def one_class(cls):
        lik = Bernoulli(shift=-3)
        lik.parameters["shift"].fix()
        return lik


class BernoulliMaximumMargin(_Likelihood):
    def __init__(self):
        super().__init__()
        c_rate = _gpr.PositiveParameter(1, 1e-3, 1e3)
        self.parameters.update({"c_rate": c_rate})
        self._all_parameters.append(c_rate)

    def log_lik(self, mu, var, y, has_value, *args, **kwargs):
        y = 2 * y - 1
        c_rate = self.parameters["c_rate"].get_value()

        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS, axis=0)

        log_density = _tf.where(
            _tf.less(_tf.math.abs(vals), 1.0),
            - _tf.math.log(1 + _tf.exp(-2 * c_rate * y * vals)),
            - _tf.math.log(1 + _tf.exp(- c_rate * y * (vals + _tf.sign(vals))))
        )
        log_density = _tf.reduce_sum(log_density * w, axis=1, keepdims=True)

        lik = _tf.reduce_sum(log_density * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS, axis=0)

        prob = self.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        prob_sims = self.cdf(sims)

        lik_var = prob * (1 - prob)
        weights = _tf.squeeze(explained_var) / (lik_var + 1e-6)
        weights = weights ** 2

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


class Dirichlet(_ContinuousLikelihood):
    def __init__(self, n_components, sharpness=1, slack=1e-6):
        super().__init__()
        self.n_components = n_components
        self.slack = slack
        self.sharpness = sharpness

    def _make_distribution(self, loc):
        return _tfd.Dirichlet(loc * self.sharpness)

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y = y * (1 - self.n_components * self.slack) + self.slack

        samples = _tf.nn.softmax(samples, axis=1)
        distribution = self._make_distribution(
            _tf.transpose(samples, [0, 2, 1])  # [data, sims, comp]
        )

        log_density = distribution.log_prob(y[:, None, :])  # [data, sims]
        log_density = _tf.reduce_mean(log_density, axis=1, keepdims=True)
        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)

        lik = _tf.reduce_sum(log_density * has_value)

        return lik

    def predict(self, mu, var, sims, explained_var,
                *args, quantiles=None, **kwargs):
        sims = _tf.nn.softmax(sims, axis=1)
        prob = _tf.reduce_mean(sims, axis=2)

        distribution = self._make_distribution(
            _tf.transpose(sims, [0, 2, 1])  # [data, sims, comp]
        )

        lik_var = distribution.variance()
        lik_var = _tf.reduce_sum(_tf.reduce_mean(lik_var, axis=1), axis=1)
        weights = _tf.reduce_sum(explained_var, axis=1) / (lik_var + 1e-6)
        weights = weights ** 2

        out = {"mean": mu,
               "variance": var,
               "probability": prob,
               "simulations": sims,
               "weights": weights}

        return out


class CategoricalGaussianIndicator(_Likelihood):
    def __init__(self, tol=1e-3, sharpness=5):
        super().__init__()
        self.tol = _tf.constant(tol, _tf.float64)
        self.sharpness = _tf.constant(sharpness, _tf.float64)

    def log_lik(self, mu, var, y, has_value, is_boundary=None,
                samples=None, *args, **kwargs):
        y = 2 * y - 1

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

        log_density = _tf.reduce_sum(log_density, axis=1, keepdims=True)

        has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)
        log_density = _tf.reduce_sum(log_density * has_value)

        return log_density * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, **kwargs):
        n_cat = _tf.shape(mu)[1]
        n_data = _tf.shape(mu)[0]

        dist = _tfd.Normal(mu, _tf.sqrt(var))
        prob_pos = dist.log_survival_function(self.tol)

        prob = _tf.nn.softmax(prob_pos, axis=1)

        log_n = _tf.math.log(_tf.cast(_tf.shape(mu)[1], _tf.float64))
        entropy = - _tf.reduce_sum(prob * _tf.math.log(prob), axis=1) / log_n
        entropy = _tf.maximum(entropy, 0.0)
        uncertainty = _tf.sqrt(_tf.reduce_mean(var, axis=1) * entropy)
        indicators = _tf.math.log(prob + 1e-6)

        idx = _tf.range(_tf.shape(mu)[0])[:, None]

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

        lik_var = prob * (1 - prob)
        lik_var = _tf.reduce_sum(lik_var, axis=1)
        weights = _tf.reduce_sum(explained_var, axis=1) / (lik_var + 1e-6)
        weights = weights ** 2

        output = {"mean": mu,
                  "variance": var,
                  "probability": prob,
                  "simulations": sims,
                  "entropy": entropy,
                  "uncertainty": uncertainty,
                  "indicators": ind_skew,
                  "weights": weights}
        return output
