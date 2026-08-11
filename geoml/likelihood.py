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

from scipy.linalg import helmert as _helmert
import copy as _copy

import geoml.warping as _warp
import geoml.parameter as _gpr
import geoml.math.tf as _tftools
import geoml.math.interpolate as _gint
import geoml.stats.probability as _gmp

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp
from scipy.stats import qmc as _qmc

_tfd = _tfp.distributions

# The rule used to integrate the likelihood noise out of a prediction when the
# warping mixes its components; see `_Likelihood._noise_nodes`. The seed is
# what keeps a scrambled sequence reproducible.
_SOBOL_NODES = 64
_SOBOL_SEED = 20260809

# How many equal-share nodes stand for the noise when a measurement is being
# described rather than integrated away; see `_Likelihood.measurement_samples`.
_MEASUREMENT_NODES = 32

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


def _aggregate(x, n_splits=None, fun=_tf.reduce_mean):
    if n_splits is None:
        return x

    # agg = _tf.stack([fun(y, axis=0) for y in _tf.split(x, n_splits, axis=0)], axis=0)
    n = _tf.cast(_tf.shape(x)[0] / n_splits, dtype=_tf.int32)
    agg = _tf.reshape(x, _tf.concat([[n_splits, n], _tf.shape(x)[1:]], axis=0))
    agg = fun(agg, axis=1)
    return agg


def _dispersion(x, n_splits=None):
    """How much a block's sub-blocks differ among themselves.

    The companion of `_aggregate`: that one takes the mean over a block's
    sub-blocks, this one their variance, over the same axis and the same
    grouping. It is the within-block dispersion -- what change of support is
    about -- and it is what says whether splitting a block would tell anyone
    anything: a block whose sub-blocks agree holds one value however finely it
    is cut.

    Read *before* `_aggregate`, and after the warping has been undone, so the
    spread is of grades rather than of the latent field. The two are not the
    same thing and only the first is reportable.

    Without a discretization the answer is *missing*, not zero: a location has
    no interior, and a block the model treats as its own centre has one it
    knows nothing about. Zero would read as "uniform inside", which is a claim
    nobody made. Divides by the number of sub-blocks, not by one less: they
    are the block, not a sample drawn from it.
    """
    if n_splits is None:
        return _tf.fill(_tf.shape(x), _tf.constant(_np.nan, dtype=x.dtype))

    n = _tf.cast(_tf.shape(x)[0] / n_splits, dtype=_tf.int32)
    grouped = _tf.reshape(
        x, _tf.concat([[n_splits, n], _tf.shape(x)[1:]], axis=0))
    return _tf.math.reduce_variance(grouped, axis=1)


def _cutoff_matrix(cutoffs, dtype):
    """Cut-offs as `(n_var, n_cutoffs)`, ready to broadcast over a block.

    One list applies to every variable and comes back with a leading 1, which
    broadcasts; a matrix is one row per variable and is left alone. Either way
    the result slots in as `cuts[:, None, :]` against a `(..., n_var, n_sim)`
    tensor.
    """
    cuts = _tf.cast(cutoffs, dtype)
    if len(cuts.shape) < 2:
        cuts = _tf.reshape(cuts, [1, -1])
    return cuts


def _proportions(x, cutoffs, n_splits=None):
    """How much of a block sits at or below each of `cutoffs`.

    The third of the reductions over a block's sub-blocks, beside `_aggregate`
    and `_dispersion`, and the one a splitting decision is made from: a share
    of 0 or 1 says the whole block is on one side of the cut-off and cutting it
    finer would find nothing, while anything in between says the block straddles
    a decision and its children would not agree.

    What counts as a cut-off is the likelihood's own business. A grade has the
    ones someone declared; an indicator has zero, that being where a category
    stops winning and its rival starts.

    Averages over the realizations as well as the sub-blocks, so a block that
    is only sometimes above the cut-off reads as partly above it. Without a
    discretization there are no sub-blocks and this is the share of the
    realizations alone -- the same number `reset_probabilities` arrives at from
    the stored simulations afterwards.

    `cutoffs` is one list for every variable, or a `(n_var, n_cutoffs)` matrix
    where they differ -- the components of a vector variable are separate
    grades and are judged against separate numbers.

    Returns `(n_blocks, n_var, n_cutoffs)`.
    """
    cuts = _cutoff_matrix(cutoffs, x.dtype)

    if n_splits is None:
        below = _tf.cast(x[..., None] <= cuts[:, None, :], x.dtype)
        return _tf.reduce_mean(below, axis=2)

    n = _tf.cast(_tf.shape(x)[0] / n_splits, dtype=_tf.int32)
    grouped = _tf.reshape(
        x, _tf.concat([[n_splits, n], _tf.shape(x)[1:]], axis=0))
    below = _tf.cast(grouped[..., None] <= cuts[:, None, :], x.dtype)
    return _tf.reduce_mean(below, axis=[1, 3])


def _divided(x, cutoffs, n_splits=None):
    """How often a block is cut in two by each of `cutoffs`.

    Not the same question as `_proportions`, and the difference is the whole
    of what refining can and cannot fix. A block whose value is uncertain
    around a cut-off has realizations either side of it, and *no* amount of
    cutting will change that -- it is the model not knowing, and the answer to
    it is another drillhole. A block whose sub-blocks disagree *within* one
    realization is a block holding two answers, and cutting is exactly what
    separates them.

    So the sub-blocks are judged one realization at a time -- is this block
    divided, in this realization? -- and only then averaged over realizations.
    Reading the share over sub-blocks and realizations together, as
    `_proportions` does, would mix the two back into one number and mark for
    splitting every block the model happens to be unsure about.

    Returns `(n_blocks, n_var, n_cutoffs)`, the share of realizations in which
    the block straddles each cut-off. Zero without a discretization: a
    location has no sub-blocks to disagree.
    """
    cuts = _cutoff_matrix(cutoffs, x.dtype)

    if n_splits is None:
        return _tf.zeros(
            _tf.concat([_tf.shape(x)[:2], [_tf.shape(cuts)[-1]]], axis=0),
            dtype=x.dtype)

    n = _tf.cast(_tf.shape(x)[0] / n_splits, dtype=_tf.int32)
    grouped = _tf.reshape(
        x, _tf.concat([[n_splits, n], _tf.shape(x)[1:]], axis=0))
    below = _tf.cast(grouped[..., None] <= cuts[:, None, :], x.dtype)

    # per realization: what share of this block's sub-blocks sit below
    share = _tf.reduce_mean(below, axis=1)
    straddles = _tf.cast(
        (share > 0.0) & (share < 1.0), x.dtype)
    return _tf.reduce_mean(straddles, axis=2)


class _Likelihood(_gpr.Parametric):

    # Whether this likelihood carries a warping, and so can say what a
    # measurement of its value would read. A categorical one cannot: its noise
    # lives in the probabilities, and there is no continuous value for a
    # sample to scatter around. Ask this rather than reaching for `warping`
    # and finding out the hard way.
    warped = False

    def __init__(self, size):
        super().__init__()
        self._size = size

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

    def _noise_nodes(self):
        """Nodes and weights that integrate the likelihood noise out.

        The noise is `eps = F^-1(u)` for `u` uniform on the unit cube -- which
        is how it was ever drawn -- so the integral over the noise density is
        an integral over that cube, and a likelihood contributes nothing to it
        but its quantile function. The two cases differ only in the array of
        nodes returned here; nothing downstream asks which one it got.

        A warping that works on each component alone needs the same node
        applied to every one of them -- so the cost does not grow with their
        number -- and Gauss-Hermite settles it in eight, agreeing with a fine
        reference to five figures where not integrating at all costs 3 to 17
        per cent of a standard deviation on the Macpass assays. A warping that
        mixes its components has to be integrated over all of them at once,
        and there scrambled Sobol reaches 0.2-0.6% of a standard deviation in
        64 points, where plain Monte Carlo of the same size gives 4-9%.
        The scramble is seeded, so the rule is fixed rather than random: a
        prediction does not depend on how it was batched, nor on any seed.

        Returns
        -------
        u : (nodes, size) in the open unit cube
        weights : (nodes,), summing to one
        """
        if self.warping.elementwise:
            u = _tfd.Normal(_tf.constant(0.0, _tf.float64),
                            _tf.constant(1.0, _tf.float64)).cdf(
                _ROOTS_8 * _np.sqrt(2.0))
            return _tf.tile(u[:, None], [1, self.size]), _WEIGHTS_8

        points = _qmc.Sobol(self.size, scramble=True,
                            seed=_SOBOL_SEED).random(_SOBOL_NODES)
        return (_tf.constant(_np.clip(points, 1e-6, 1 - 1e-6), _tf.float64),
                _tf.fill([_SOBOL_NODES],
                         _tf.constant(1 / _SOBOL_NODES, _tf.float64)))

    def _measurement_nodes(self, n_nodes):
        """Nodes that *represent* the noise instead of integrating against it.

        A different job from `_noise_nodes`, and so a different rule. Gauss-
        Hermite is built to make an integral exact, and pays for it with a few
        far-flung points carrying weights of 1e-4: excellent for a mean,
        useless as a picture of a distribution. Here every node stands for the
        same share of the probability, so the values they produce can be
        pooled and read as a sample -- quantiles and all -- with no weights
        anywhere. In more than one dimension the equal-share set is a
        scrambled Sobol sequence, which is equal-weight by construction.
        """
        if self.warping.elementwise:
            u = (_np.arange(n_nodes) + 0.5) / n_nodes
            return _tf.constant(_np.tile(u[:, None], [1, self.size]),
                                _tf.float64)

        points = _qmc.Sobol(self.size, scramble=True,
                            seed=_SOBOL_SEED).random(n_nodes)
        return _tf.constant(_np.clip(points, 1e-6, 1 - 1e-6), _tf.float64)

    def _noise_values(self):
        """The noise nodes with the quantile applied: `(eps, weights)`.

        `eps` is `(nodes, size, 1)` in warped space, `weights` sums to one.
        The single override point for a likelihood whose noise is not one
        distribution: a mixture integrates each mechanism on its own nodes
        and weights them, which is exact where a joint quantile would need
        root-finding.
        """
        u, weights = self._noise_nodes()
        dist = self._make_distribution(_tf.constant(0.0, dtype=_tf.float64))
        return dist.quantile(u[:, :, None]), weights

    def _measurement_values(self, n_nodes):
        """Equal-share noise values standing for a fresh measurement."""
        u = self._measurement_nodes(n_nodes)
        dist = self._make_distribution(_tf.constant(0.0, dtype=_tf.float64))
        return dist.quantile(u[:, :, None])

    def measurement_samples(self, sims, n_nodes=_MEASUREMENT_NODES):
        """What a *measurement* at each location would read.

        A prediction reports the ground, the noise having been integrated out,
        so its simulations are intervals for a quantity no assay ever
        observes. This keeps the node values instead of averaging them, which
        is the same computation stopped one step earlier: `n_sim * n_nodes`
        equally likely values per location, the exact predictive distribution
        of a sample. It is what any comparison against measured data needs --
        an accuracy plot, a cross-validation -- and it is meant for the few
        thousand locations that carry measurements, never for a block model.

        Returns
        -------
        (rows, variables, n_sim * n_nodes)
        """
        noise = self._measurement_values(n_nodes)

        return _tf.concat(
            [self._back_transform(sims + noise[i][None])
             for i in range(n_nodes)], axis=2)

    def _back_transform(self, sims):
        """Simulations out of the latent space, one realization at a time."""
        sims = _tf.transpose(sims, [2, 0, 1])
        sims = _tf.map_fn(lambda x: self.warping.backward(x), sims)
        return _tf.transpose(sims, [1, 2, 0])

    def _values_and_noise(self, sims, include_noise):
        """What a prediction reports, and how far a sample of it would fall.

        Without the integration there is no noise variance to report: the
        answer is *missing* rather than zero, as it is for `_dispersion` where
        a location has no interior. Zero would claim that a measurement here
        is exact, which is not something anyone said.
        """
        if include_noise:
            return self.integrated_backward(sims)
        values = self._back_transform(sims)
        return values, _tf.fill(_tf.shape(values),
                                _tf.constant(_np.nan, values.dtype))

    def integrated_backward(self, sims):
        """Simulations out of the latent space, with the noise integrated out.

        Reports `E[g(z + eps)]` rather than `g(z + eps)` for some drawn `eps`:
        the value the ground would show once the measurement error and the
        variability below the model's resolution are averaged over. There is
        no point in mapping noise, and a block of any size integrates it away
        anyway -- which is the conceptual difference from a conventional
        geostatistical simulation, where signal and noise are conflated.

        The second moment comes off the same nodes for nothing, and answers a
        different question: how far a fresh *measurement* of this value would
        scatter. The first is what the ground holds, the second what a sample
        of it would read.

        The nodes are consumed one at a time, so the largest tensor in the
        pipeline is never copied: the cost is in time, not in memory.

        Returns
        -------
        mean : the integrated value, in the variable's own units
        variance : the spread of a measurement of it, same units
        """
        noise, weights = self._noise_values()

        blank = _tf.zeros(
            [_tf.shape(sims)[0], self.warping.size_in, _tf.shape(sims)[2]],
            dtype=sims.dtype)

        def accumulate(carry, node):
            eps, weight = node
            value = self._back_transform(sims + eps[None])
            return (carry[0] + weight * value,
                    carry[1] + weight * value ** 2)

        mean, second = _tf.foldl(accumulate, (noise, weights),
                                 initializer=(blank, blank))
        return mean, _tf.maximum(second - mean ** 2, 0.0)

    def initialize(self, y):
        pass


class _ContinuousLikelihood(_Likelihood):
    """One continuous likelihood, whatever the number of components.

    A scalar variable and a vector one compute the same thing on vectors of
    different sizes, so one class serves both: `size` is the warping's, and
    the two places the old scalar/multivariate split actually differed are
    both decided by `warping.elementwise` -- the same flag `_noise_nodes`
    already reads. The expectation in `log_lik` is Gauss-Hermite quadrature,
    exact per component, falling back to Monte Carlo over the latent samples
    when the warping mixes its components; and a row with a missing
    component is dropped whole only in that same case (a mix spreads the
    hole over every warped column, and returns one log-derivative for the
    row rather than one per component).
    """
    warped = True

    def __init__(self, warping=None, sharpness=1):
        """
        Initializer for continuous likelihoods.

        Parameters
        ----------
        warping : geoml.warping.Warping
            A Warping object that normalizes the data values. Its output size
            is the number of components modelled.
        """
        if warping is None:
            warping = _warp.ZScore(1)
        super().__init__(warping.size_out)
        self.warping = self._register(warping)
        self.sharpness = sharpness

    def initialize(self, y):
        self.warping.initialize(y)

    def log_lik(self, mu, var, y, has_value, samples=None,
                *args, **kwargs):
        y_warped, log_derivative = self.warping.forward(y)

        if self.size > 1:
            # the log-derivative comes back per row, and a mixing warping
            # spreads a missing component over every warped one: the row is
            # weighed whole
            has_value = _tf.reduce_mean(has_value, axis=1, keepdims=True)

        if not self.warping.elementwise:
            distribution = self._make_distribution(samples)

            log_density = distribution.log_prob(y_warped[:, :, None])
            log_density = _tf.math.reduce_mean(
                log_density, axis=2, keepdims=False)

        else:
            vals = _ROOTS_64[None, None, :]
            vals = _tf.sqrt(2 * var[:, :, None]) * vals + mu[:, :, None]  # [n_data, size, n_vals]
            w = _WEIGHTS_64[None, None, :]

            distribution = self._make_distribution(vals)

            log_density = distribution.log_prob(y_warped[:, :, None])
            log_density = _tf.reduce_sum(log_density * w, axis=2, keepdims=False)

        lik = _tf.reduce_sum(log_density * has_value) \
              + _tf.reduce_sum(log_derivative[:, None] * has_value)

        return lik * self.sharpness

    def predict(self, mu, var, sims, explained_var, *args, include_noise=True,
                n_splits=None, cutoffs=None, **kwargs):
        # One field, and everything is read from it. The noise is integrated
        # out rather than drawn, so what comes back is already free of the
        # part of the spread that refining cannot resolve: a block's
        # dispersion is the ground's, and a block straddling a cut-off really
        # does straddle it.
        values, noise = self._values_and_noise(sims, include_noise)

        # taken from the sub-blocks, so before they are averaged away; one
        # value per realization, then the mean over them, which is the
        # dispersion of a block's interior as the model sees it; each
        # component of a vector variable on its own account
        dispersion = _tf.reduce_mean(
            _dispersion(values, n_splits=n_splits), axis=2)
        noise = _aggregate(_tf.reduce_mean(noise, axis=2), n_splits=n_splits)

        sims = _aggregate(values, n_splits=n_splits)
        mu = _aggregate(mu, n_splits=n_splits)
        var = _aggregate(var, n_splits=n_splits)

        avg_sim = _tf.reduce_mean(sims, axis=2)
        out = {"mean": mu,
               "variance": var,
               "simulations": sims,
               "average_sim": avg_sim,
               "dispersion": dispersion,
               "noise_variance": noise,
               "uncertainty": _tf.reduce_mean(var, axis=1),
               }
        if cutoffs is not None:
            out["proportions"] = _proportions(
                values, cutoffs, n_splits=n_splits)
            out["divided"] = _divided(
                values, cutoffs, n_splits=n_splits)
        return out

    def _make_distribution(self, *args, **kwargs):
        raise NotImplementedError


class Gaussian(_ContinuousLikelihood):
    """
    Gaussian likelihood.

    Equivalent to a squared error model. The latent variable maps to the mean,
    while the noise variance is a parameter.
    """
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "noise",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 0.1,
                _np.ones([1, self.size, 1]) * 1e-6,
                _np.ones([1, self.size, 1]) * 10
            )
        )

    def _make_distribution(self, loc):
        return _tfd.Normal(loc, _tf.sqrt(self.parameters["noise"].get_value()))


class Laplace(_ContinuousLikelihood):
    """
    Laplace's likelihood.

    Equivalent to a linear error model. The latent variable maps to the mean,
    while the distribution's scale factor is a parameter.
    """
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "scale",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 0.1,
                _np.ones([1, self.size, 1]) * 1e-12,
                _np.ones([1, self.size, 1]) * 10
            )
        )

    def _make_distribution(self, loc):
        return _tfd.Laplace(loc, self.parameters["scale"].get_value())


class Gamma(_ContinuousLikelihood):
    """
    Gamma likelihood.

    Used for strictly positive variables. The latent variable is shifted by a
    parameter and then mapped to the distribution's shape. The rate parameter
    is fixed at 1.0.
    """
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "mean_alpha",
            _gpr.RealParameter(
                _np.zeros([1, self.size, 1]),
                _np.zeros([1, self.size, 1]) - 3,
                _np.zeros([1, self.size, 1]) + 3
            )
        )

    def _make_distribution(self, loc):
        mean_alpha = self.parameters["mean_alpha"].get_value()
        return _tfd.Gamma(_tf.exp(loc + mean_alpha) + 0.01,
                          _tf.constant(1.0, _tf.float64))


class StudentT(_ContinuousLikelihood):
    """
    Student-T likelihood.

    A heavy-tailed distribution. The latent variable maps to the mean,
    while the scale and degrees of freedom are parameters.
    """
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "scale",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 0.1,
                _np.ones([1, self.size, 1]) * 1e-9,
                _np.ones([1, self.size, 1]) * 10
            )
        )
        self._add_parameter(
            "df",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 5.0,
                _np.ones([1, self.size, 1]) * 2.01,
                _np.ones([1, self.size, 1]) * 50.0
            )
        )

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
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "epsilon",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 0.001,
                _np.ones([1, self.size, 1]) * 1e-9,
                _np.ones([1, self.size, 1]) * 10
            )
        )
        self._add_parameter(
            "c_rate",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 1,
                _np.ones([1, self.size, 1]) * 1e-3,
                _np.ones([1, self.size, 1]) * 1e3
            )
        )

    def _make_distribution(self, loc):
        return _gmp.EpsilonInsensitive(
            loc,
            scale=self.parameters["c_rate"].get_value(),
            epsilon=self.parameters["epsilon"].get_value()
        )


class Huber(_ContinuousLikelihood):
    """
    Huber's likelihood.

    Based on the Huber loss.
    """
    def __init__(self, warping=None, sharpness=1):
        super().__init__(warping, sharpness)
        self._add_parameter(
            "threshold",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 3,
                _np.ones([1, self.size, 1]) * 1e-2,
                _np.ones([1, self.size, 1]) * 100)
        )
        self._add_parameter(
            "std",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 1,
                _np.ones([1, self.size, 1]) * 1e-3,
                _np.ones([1, self.size, 1]) * 10)
        )

    def _make_distribution(self, loc):
        return _gmp.Huber(
            loc,
            scale=self.parameters["std"].get_value(),
            epsilon=self.parameters["threshold"].get_value()
        )


# The multivariate twins are the scalar likelihoods with a wider default
# warping -- the machinery is one class since the scalar/multivariate split
# collapsed into `_ContinuousLikelihood`. `MultivariateLaplace` and
# `MultivariateHuber` keep their historical parameter names and initial
# values, so a model saved with either still loads.

class MultivariateGaussian(Gaussian):
    def __init__(self, n_components, warping=None, sharpness=1):
        if warping is None:
            warping = _warp.ZScore(n_components)
        super().__init__(warping, sharpness=sharpness)


class MultivariateLaplace(_ContinuousLikelihood):
    def __init__(self, n_components, warping=None, sharpness=1):
        if warping is None:
            warping = _warp.ZScore(n_components)
        super().__init__(warping, sharpness=sharpness)
        self._add_parameter(
            "rate",
            _gpr.PositiveParameter(
                _np.ones([1, self.size, 1]) * 0.1,
                _np.ones([1, self.size, 1]) * 1e-6,
                _np.ones([1, self.size, 1]) * 10)
        )

    def _make_distribution(self, loc):
        return _tfd.Laplace(loc, self.parameters["rate"].get_value())


class MultivariateEpsilonInsensitive(EpsilonInsensitive):
    def __init__(self, n_components, warping=None, sharpness=1):
        if warping is None:
            warping = _warp.ZScore(n_components)
        super().__init__(warping, sharpness=sharpness)


class MultivariateHuber(Huber):
    def __init__(self, n_components, warping=None, sharpness=1):
        if warping is None:
            warping = _warp.ZScore(n_components)
        super().__init__(warping, sharpness=sharpness)
        self.parameters["std"].set_value(_np.ones([1, self.size, 1]) * 0.1)


class _MixtureDensity:
    """The density of a weighted mixture, for the training expectation.

    Not a TFP distribution: `log_lik` asks for nothing but `log_prob`, and
    the prediction side never touches this object -- the noise integral runs
    each component on its own nodes (`Mixture._noise_values`), which is exact
    where a joint quantile would need root-finding.
    """

    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = weights

    def log_prob(self, x):
        log_w = _tf.math.log(self.weights)
        parts = _tf.stack([d.log_prob(x) for d in self.distributions], axis=0)
        return _tf.reduce_logsumexp(
            parts + log_w[:, None, None, None], axis=0)


class Mixture(_ContinuousLikelihood):
    """A likelihood whose noise comes from one of several mechanisms.

    Each measurement is drawn from one of the `components` -- continuous
    likelihoods of any kind, sharing the latent location -- with trainable
    proportions. The classic use is two: a narrow one for the natural
    short-range variability and a wide one for contaminated measurements,
    which a single number cannot tell apart. A heavy-tailed likelihood
    downweights an outlier; a mixture also *names* it (`responsibilities`),
    says how often it happens (the weight), and keeps it out of the ground.

    Every component after the first is taken to describe contamination
    unless `contamination` says otherwise. What that means sits entirely at
    prediction: `log_lik` and `measurement_samples` use the full mixture --
    training must explain the data as it is, and a fresh assay can be a bad
    one -- while `integrated_backward` averages the prediction over the
    genuine components alone, because a contaminated reading replaces the
    measurement and says nothing about the ground it displaced. On a
    nonlinear warping the difference is not academic: integrating the wide
    component into a block value biases it the way the contamination is
    skewed (measured at +6 to +17% on a lognormal-like synthetic case).

    The component choice applies to a row as a whole -- a bad sample is bad
    across its columns -- so the weights are one simplex, not one per
    column. A component's own warping is inert (the mixture's is the one
    applied, once) and its parameters are fixed on registration.

    Initialize the contamination component *wide*: its job is to be the
    cheaper explanation for a tail value before a trainable warping bends
    itself to accommodate one. Guard the other flank of the same fight with
    `ZScore(size, robust=True)` in the warping: it initializes on a
    winsorized copy of the data, so a gross outlier cannot set the scale
    everything else is normalized by. Measured together, the two carried
    the pathological contamination draw from 4.3x the clean-data error to
    1.3x.
    """

    def __init__(self, components, warping, weights=None, contamination=None,
                 sharpness=1):
        """
        Parameters
        ----------
        components : list of _ContinuousLikelihood
            The noise mechanisms, of the same kind or not. A component's
            parameters are `[1, size, 1]`-shaped and broadcast, so scalar
            components serve a mixture of any width -- one noise per
            mechanism across the columns -- while sized components keep a
            parameter per column. Their own warpings are ignored.
        warping : geoml.warping.Warping
            The mixture's own warping, applied once to the data. Required:
            the components' warpings play no part, so this is the one place
            the mixture's size can honestly come from.
        weights : array-like, optional
            Initial mixing proportions, one per component, summing to one.
            Default: 0.95 on the first, the rest split evenly.
        contamination : list of bool, optional
            Which components describe error rather than ground. Default:
            every one but the first. At least one component must be genuine.
        """
        components = list(components)
        if len(components) < 2:
            raise ValueError("a mixture needs at least two components")
        super().__init__(warping, sharpness)

        for component in components:
            # the component's own warping plays no part; freeze it so its
            # parameters do not drift in training
            for parameter in component.warping._all_parameters:
                parameter.fix()
        self.components = [self._register(c) for c in components]

        if contamination is None:
            contamination = [False] + [True] * (len(components) - 1)
        contamination = [bool(c) for c in contamination]
        if len(contamination) != len(components):
            raise ValueError("one contamination flag per component")
        if all(contamination):
            raise ValueError("at least one component must describe the "
                             "ground rather than contamination")
        self.contamination = contamination

        if weights is None:
            n = len(components)
            weights = _np.full([n], 0.05 / (n - 1))
            weights[0] = 0.95
        self._add_parameter(
            "weights",
            _gpr.CompositionalParameter(_np.asarray(weights, dtype=float)))

    def _component_distributions(self):
        zero = _tf.constant(0.0, _tf.float64)
        return [c._make_distribution(zero) for c in self.components]

    def _make_distribution(self, loc):
        w = self.parameters["weights"].get_value()
        return _MixtureDensity(
            [c._make_distribution(loc) for c in self.components], w)

    def _noise_values(self):
        """The genuine components' nodes, weighted by their share.

        The contamination components are left out and the remaining weights
        renormalized: a contaminated reading replaces the measurement, so it
        has no place in the average that says what the ground holds.
        """
        u, node_w = self._noise_nodes()
        w = self.parameters["weights"].get_value()
        keep = [i for i, bad in enumerate(self.contamination) if not bad]
        w_kept = _tf.gather(w, keep)
        w_kept = w_kept / _tf.reduce_sum(w_kept)

        distributions = self._component_distributions()
        eps = _tf.concat(
            [distributions[i].quantile(u[:, :, None]) for i in keep], axis=0)
        weights = _tf.concat(
            [node_w * w_kept[j] for j in range(len(keep))], axis=0)
        return eps, weights

    def _measurement_values(self, n_nodes):
        """Equal-share nodes of the full mixture, contamination included.

        A fresh measurement can be a bad one, so what a sample would read is
        described by everything. The mixture quantile has no closed form;
        sixty bisections of the closed-form CDF settle it to working
        precision, once per trace.
        """
        u = self._measurement_nodes(n_nodes)[:, :, None]
        w = self.parameters["weights"].get_value()
        distributions = self._component_distributions()

        quantiles = _tf.stack([d.quantile(u) for d in distributions], axis=0)
        lo = _tf.reduce_min(quantiles, axis=0)
        hi = _tf.reduce_max(quantiles, axis=0)

        def mixture_cdf(x):
            parts = _tf.stack([d.cdf(x) for d in distributions], axis=0)
            return _tf.reduce_sum(parts * w[:, None, None, None], axis=0)

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = mixture_cdf(mid) < u
            lo = _tf.where(below, mid, lo)
            hi = _tf.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def responsibilities(self, latent_mean, latent_variance, values):
        """How likely each row is to have come from each component.

        The mixture's unique diagnostic: a heavy-tailed likelihood can
        absorb an outlier, only a mixture can point at it. Row-wise, since
        the component choice applies to the row as a whole.

        Parameters
        ----------
        latent_mean, latent_variance : (n,) or (n, size)
            The model's posterior at the measured locations, as `predict`
            stores them.
        values : (n,) or (n, size)
            The measurements, in their own units.

        Returns
        -------
        (n, n_components), rows summing to one.
        """
        mu = _tf.constant(_np.atleast_2d(_np.transpose(latent_mean)).T,
                          _tf.float64)
        var = _tf.constant(_np.atleast_2d(_np.transpose(latent_variance)).T,
                           _tf.float64)
        y = _tf.constant(_np.atleast_2d(_np.transpose(values)).T, _tf.float64)
        y_warped, _ = self.warping.forward(y)

        vals = _tf.sqrt(2 * var[:, :, None]) * _ROOTS_64[None, None, :] \
            + mu[:, :, None]
        log_gh = _tf.math.log(_WEIGHTS_64)[None, None, :]

        # E_q[p_k(y | f)] per column by Gauss-Hermite, joined over columns
        # (exact: the density factorizes and the marginals are independent),
        # then weighted across components
        log_lik = []
        for component in self.components:
            distribution = component._make_distribution(vals)
            log_density = distribution.log_prob(y_warped[:, :, None])
            per_column = _tf.reduce_logsumexp(log_density + log_gh, axis=2)
            log_lik.append(_tf.reduce_sum(per_column, axis=1))
        log_lik = _tf.stack(log_lik, axis=1)

        log_w = _tf.math.log(self.parameters["weights"].get_value())
        log_post = log_lik + log_w[None, :]
        return _tf.exp(log_post
                       - _tf.reduce_logsumexp(log_post, axis=1,
                                              keepdims=True)).numpy()


class Bernoulli(_Likelihood):
    def __init__(self, shift=0, sharpness=1):
        """
        Bernoulli's likelihood.

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

    def predict(self, mu, var, sims, explained_var, n_splits=None, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        shift = self.parameters["shift"].get_value()
        slope = self.parameters["slope"].get_value()
        # distribution = _tfd.Normal(- shift, _tf.constant(1.0, _tf.float64))
        distribution = _tfd.Normal(- shift, 1 / slope)

        prob = distribution.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        prob = _aggregate(prob, n_splits)
        mu = _aggregate(mu, n_splits)
        var = _aggregate(var, n_splits)
        explained_var = _aggregate(explained_var, n_splits)

        entropy = (- prob * _tf.math.log(prob)
                   - (1 - prob) * _tf.math.log(1 - prob)) / _np.log(2)
        uncertainty = _tf.sqrt(_tf.squeeze(var) * entropy)

        prob_sims = distribution.cdf(sims)
        prob_sims = _aggregate(prob_sims, n_splits)

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
        lik = cls(shift=-3, sharpness=sharpness)
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

    def predict(self, mu, var, sims, explained_var, n_splits=None, *args, **kwargs):
        vals = _tf.expand_dims(_ROOTS_64, axis=0)
        vals = _tf.sqrt(2 * var) * vals + mu  # [n_data, n_vals]
        w = _tf.expand_dims(_WEIGHTS_64, axis=0)

        prob = self.cdf(vals)
        prob = _tf.reduce_sum(prob * w, axis=1)

        prob = _aggregate(prob, n_splits)
        mu = _aggregate(mu, n_splits)
        var = _aggregate(var, n_splits)

        entropy = (- prob * _tf.math.log(prob)
                   - (1 - prob) * _tf.math.log(1 - prob)) / _np.log(2)
        uncertainty = _tf.sqrt(_tf.squeeze(var) * entropy)

        prob_sims = self.cdf(sims)
        prob_sims = _aggregate(prob_sims, n_splits)

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

    def cdf(self, x):
        c_rate = self.parameters["c_rate"].get_value()
        prob = _tf.where(
            _tf.less(_tf.math.abs(x), 1.0),
            1 / (1 + _tf.exp(-2 * c_rate * x)),
            1 / (1 + _tf.exp(- c_rate * (x + _tf.sign(x))))
        )
        return prob


class _CategoricalLikelihood(_Likelihood):
    def __init__(self, size):
        super().__init__(size)

    @staticmethod
    def entropy_and_indicators(probabilities, var, explained_var):
        n_cat = _tf.shape(probabilities)[1]
        log_n = _tf.math.log(_tf.cast(n_cat, _tf.float64))

        entropy = - _tf.reduce_sum(
            probabilities * _tf.math.log(probabilities + 1e-6), axis=1) / log_n
        entropy = _tf.maximum(entropy, 0.0)
        avg_var = _tf.reduce_sum(var * probabilities, axis=1)
        uncertainty = _tf.sqrt(avg_var * entropy)
        indicators = _tf.math.log(probabilities + 1e-6)

        # A category's rival is the best of the others, which is the runner-up
        # for whoever wins and the winner for everybody else -- two row maxima,
        # the second taken with the winners dropped, rather than one full-size
        # scatter per category. The tie is what the count is for: two
        # categories sharing the maximum are each other's rival, so both come
        # out at zero and the contact stays the zero level set.
        best = _tf.reduce_max(indicators, axis=1, keepdims=True)
        winner = indicators >= best
        runner_up = _tf.reduce_max(
            _tf.where(winner, indicators.dtype.min, indicators),
            axis=1, keepdims=True)
        shared = _tf.reduce_sum(_tf.cast(winner, _tf.float64), axis=1,
                                keepdims=True) > 1.0
        ind_skew = indicators - _tf.where(
            winner, _tf.where(shared, best, runner_up), best)

        lik_var = probabilities * (1 - probabilities)
        lik_var = _tf.reduce_sum(lik_var, axis=1)
        weights = _tf.reduce_sum(explained_var, axis=1) / (lik_var + 1e-6)

        return entropy, uncertainty, ind_skew, weights

    def _resolve(self, prob, var, explained_var, n_splits):
        """The indicator picture, read from the sub-blocks and then averaged.

        Taken before the sub-blocks are aggregated, so a block holding a
        contact is described by the sub-blocks falling either side of it
        rather than by the mixed probability their average comes to. The two
        are not the same and the second is the less useful: averaged first, a
        block half granite and half schist looks like a place where the model
        cannot decide, instead of one where it decides differently in
        different corners.

        `ind_skew` is a category's log-odds against its best rival, so it is
        positive where that category wins and zero exactly where two are tied.
        The contact is its zero level set, which is what makes the share of a
        block on either side of zero the same question a grade asks of a
        cut-off -- and `proportions` the share of the block each category
        holds, which is worth having in its own right on a domained model.
        """
        entropy, uncertainty, ind_skew, weights = self.entropy_and_indicators(
            prob, var, explained_var)

        # `_proportions` counts what is at or *below* the cut-off, and a
        # category holds the ground where its skew is above zero. There is no
        # realization axis here -- the probabilities are already expectations
        # -- so the dummy one makes `_divided` degenerate to the same test:
        # do this block's sub-blocks disagree about who holds it.
        skew = ind_skew[:, :, None]
        share = 1.0 - _proportions(skew, [0.0], n_splits=n_splits)[:, :, 0]
        divided = _divided(skew, [0.0], n_splits=n_splits)[:, :, 0]

        return {"entropy": _aggregate(entropy, n_splits),
                "uncertainty": _aggregate(uncertainty, n_splits),
                "indicators": _aggregate(ind_skew, n_splits),
                "weights": _aggregate(weights, n_splits),
                "proportions": share,
                "divided": divided}


class CategoricalGaussianIndicator(_CategoricalLikelihood):
    """
    Gaussian likelihood for indicator variables.

    Assumes mutually exclusive categories (i.e. no geological rules),
    leading to maximum entropy far from the data points. Is capable of
    dealing with boundary data.
    """
    def __init__(self, n_components, tol=1e-3, sharpness=1):
        """
        Initializer for CategoricalGaussianIndicator.

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
        super().__init__(n_components)
        self.tol = _tf.constant(tol, _tf.float64)
        self.sharpness = _tf.constant(sharpness, _tf.float64)

    def log_lik(self, mu, var, y, has_value, is_boundary=None,
                samples=None, *args, **kwargs):
        y = 2 * y - 1

        # if self._use_monte_carlo:
        #     # pos = _tf.where(
        #     #     _tf.greater(samples, self.tol),
        #     #     _tf.ones_like(samples),
        #     #     _tf.zeros_like(samples)
        #     # )
        #     # neg = _tf.where(
        #     #     _tf.less(samples, - self.tol),
        #     #     _tf.ones_like(samples),
        #     #     _tf.zeros_like(samples)
        #     # )
        #     #
        #     # prob_pos = _tf.reduce_mean(pos, axis=-1)
        #     # prob_neg = _tf.reduce_mean(neg, axis=-1)
        #     # prob_zero = 1 - prob_neg - prob_pos
        #     #
        #     # prob_pos = _tf.math.log(prob_pos + 1e-6)
        #     # prob_neg = _tf.math.log(prob_neg + 1e-6)
        #     # prob_zero = _tf.math.log(prob_zero + 1e-6)
        #
        #     dist = _tfd.Normal(samples, 1.0)  #self.tol)
        #     prob_neg = dist.log_cdf(- self.tol)
        #     prob_zero = _tf.math.log(
        #         dist.cdf(self.tol) - dist.cdf(- self.tol) + 1e-6)
        #     prob_pos = dist.log_survival_function(self.tol)
        #
        #     # prob_neg = _tf.reduce_mean(prob_neg, axis=-1)
        #     # prob_pos = _tf.reduce_mean(prob_pos, axis=-1)
        #     # prob_zero = _tf.reduce_mean(prob_zero, axis=-1)
        #
        #     n_sim = _tf.shape(samples)[2]
        #     y = _tf.tile(y[:, :, None], [1, 1, n_sim])
        #
        #     log_density = _tf.where(
        #         _tf.less(y, - self.tol),
        #         prob_neg,
        #         _tf.where(_tf.greater(y, self.tol),
        #                   prob_pos,
        #                   prob_zero)
        #     )
        #     log_density = _tf.reduce_mean(log_density, axis=2)

        # else:
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

    def predict(self, mu, var, sims, explained_var, n_splits=None, *args, **kwargs):
        # n_cat = _tf.shape(mu)[1]
        # n_data = _tf.shape(mu)[0]

        dist = _tfd.Normal(mu, _tf.sqrt(var))
        log_prob_positive = dist.log_survival_function(self.tol)
        log_prob_negative = dist.log_prob(- self.tol)

        # probability of being class i AND not being the others -- the whole
        # row of negative log-probabilities, with category i's own swapped
        # for its positive one
        log_prob_final = (
            _tf.reduce_sum(log_prob_negative, axis=1, keepdims=True)
            - log_prob_negative + log_prob_positive)

        # prob = _tf.nn.softmax(log_prob_positive, axis=1)
        prob = _tf.nn.softmax(log_prob_final, axis=1)

        indicators = self._resolve(prob, var, explained_var, n_splits)

        prob = _aggregate(prob, n_splits)
        mu = _aggregate(mu, n_splits)
        var = _aggregate(var, n_splits)
        explained_var = _aggregate(explained_var, n_splits)
        sims = _aggregate(sims, n_splits)

        output = {"mean": mu,
                  "variance": var,
                  "probability": prob,
                  "simulations": sims}
        output.update(indicators)
        return output


class HierarchicalGaussianIndicator(CategoricalGaussianIndicator):
    """
    Gaussian likelihood for indicator variables with geological rules.

    Assumes a priority order among categories, so that a point at which a higher priority category is positive will
    automatically be negative for the lower priority ones. This allows the modelling of intrusions by giving a high
    priority to the intruding rock, and depositions by giving a low priority to the deposited layer, making it
    conform to the geometry of the rocks below it.

    The priority is defined by the order of the labels in the data object, from lowest to highest.
    """

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

    def predict(self, mu, var, sims, explained_var, n_splits=None, *args, **kwargs):
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

        # probability of being class i AND not being the others -- the whole
        # row of negative log-probabilities, with category i's own swapped
        # for its positive one
        log_prob_final = (
            _tf.reduce_sum(log_prob_negative, axis=1, keepdims=True)
            - log_prob_negative + log_prob_positive)

        prob = _tf.nn.softmax(log_prob_final, axis=1)

        indicators = self._resolve(prob, var, explained_var, n_splits)

        prob = _aggregate(prob, n_splits)
        mu = _aggregate(mu, n_splits)
        var = _aggregate(var, n_splits)

        output = {"mean": mu,
                  "variance": var,
                  "probability": prob,
                  "simulations": sims}
        output.update(indicators)
        return output


class OrderedGaussianIndicator(_CategoricalLikelihood):
    """
    Gaussian likelihood for indicator variables of conformable layers.

    By assuming conformable layers, it is possible to model multiple categories with a single latent variable. The
    thresholds that define the contacts are determined during training. It is useful to add a linear trend to the
    network's output.
    """
    def __init__(self, levels, tol=1e-6, sharpness=1):
        """
        Initializer for OrderedGaussianIndicator.

        Parameters
        ----------
        levels : int
            Number of conformable surfaces, one less than the number of rock layers.
        tol : double
            Normal score tolerance for boundary data.
        sharpness : int
            Data augmentation. The weight of the data is multiplied by this
            factor. Results in sharper transitions between categories.
        """
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

    def predict(self, mu, var, sims, explained_var, n_splits=None, *args, **kwargs):
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

        indicators = self._resolve(prob, var, explained_var, n_splits)

        prob = _aggregate(prob, n_splits)

        # if self.levels > 1:
        #     mu = mu - thresholds[None, :]
        #     sims = sims - thresholds[None, :, None]

        mu = _aggregate(mu, n_splits)
        var = _aggregate(var, n_splits)

        output = {"mean": _tf.tile(mu, [1, self.levels + 1]),
                  "variance": _tf.tile(var, [1, self.levels + 1]),
                  "simulations": _tf.tile(sims, [1, self.levels + 1, 1]),
                  "probability": prob}
        output.update(indicators)
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
