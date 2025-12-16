# geoML - machine learning models for geospatial data
# Copyright (C) 2025  Ítalo Gomes Gonçalves
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

import geoml.tftools as _tftools
import geoml.interpolation as _gint

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp

_tfd = _tfp.distributions


class EpsilonInsensitive(_tfd.Distribution):
    """
    A custom implementation of the Epsilon Insensitive distribution (Gonçalves et al., 2022).
    """

    def __init__(self,
                 loc,
                 scale,
                 epsilon,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="EpsilonInsensitive"):
        """
        Initialize the distribution.

        Args:
          loc: The mean (mu) of the distribution.
          scale: The rate (c) of the distribution.
          epsilon: The insensitivity parameter.
        """
        with _tf.name_scope(name) as name:
            # Convert inputs to Tensors and store them as private attributes.
            # This handles Python floats, lists, NumPy arrays, etc.
            self._loc = _tf.convert_to_tensor(loc, name='loc')
            self._scale = _tf.convert_to_tensor(scale, name='scale')
            self._epsilon = _tf.convert_to_tensor(epsilon, name='epsilon')

            # Infer the dtype from the parameters.
            # TFP distributions must have a single dtype.
            dtype = self._loc.dtype

            # Call the parent constructor.
            super(EpsilonInsensitive, self).__init__(
                dtype=dtype,
                reparameterization_type=_tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name)

    # 2. Define public parameter properties

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the standard deviation."""
        return self._scale

    @property
    def epsilon(self):
        """Distribution parameter for the insensitivity."""
        return self._epsilon

    @property
    def parameters(self):
        """Returns a dict of parameters."""
        return dict(loc=self.loc, scale=self.scale, epsilon=self.epsilon)

    # 3. Define shape methods

    def _batch_shape_tensor(self):
        # The batch shape is the broadcasted shape of loc and scale.
        # e.g., if loc.shape=[2, 1] and scale.shape=[1, 3],
        # the batch_shape is [2, 3].
        shape = _tf.broadcast_dynamic_shape(
            _tf.shape(self.loc), _tf.shape(self.scale)
        )
        shape = _tf.broadcast_dynamic_shape(shape, _tf.shape(self.epsilon))
        return shape

    def _event_shape_tensor(self):
        # A single sample from a univariate Normal is a scalar.
        # A scalar has an empty shape, represented as [].
        return _tf.constant([], dtype=_tf.int32)

    # 4. Define core logic (_log_prob and _sample_n)

    def _log_prob(self, value):
        # Convert value to the correct dtype, just in case.
        value = _tf.convert_to_tensor(value, dtype=self.dtype)

        z = _tf.math.abs(value - self.loc)

        log_density = _tf.where(
            _tf.less_equal(z, self.epsilon),
            _tf.zeros_like(z),
            - self.scale * (z - self.epsilon)
        )
        log_density = log_density - _tf.math.log(
            2 * (self.epsilon + 1 / self.scale))
        return log_density

    def _cdf(self, value):
        z = value - self.loc

        val_1 = _tf.math.exp(self.scale * (z + self.epsilon)) / self.scale
        val_2 = 1.0 / self.scale + z + self.epsilon
        val_3 = 2 * self.epsilon + 2 / self.scale \
                * (1 - 0.5 * _tf.math.exp(- self.scale * (z - self.epsilon)))

        prob = _tf.where(_tf.greater(z, - self.epsilon), val_2, val_1)
        prob = _tf.where(_tf.greater(z, self.epsilon), val_3, prob)

        prob = prob * 0.5 / (self.epsilon + 1 / self.scale)
        return prob

    def _quantile(self, p):
        area = 2 * (self.epsilon + 1 / self.scale)
        p_1 = (1 / self.scale) / area
        p_2 = p_1 + 2 * self.epsilon / area

        val_1 = _tf.math.log(area * self.scale * p) / self.scale - self.epsilon
        val_2 = area * p - 1 / self.scale - self.epsilon
        val_3 = self.epsilon - _tf.math.log(2 - self.scale * (area * p - 2 * self.epsilon)) / self.scale

        x = _tf.where(_tf.greater(p, p_1), val_2, val_1)
        x = _tf.where(_tf.greater(p, p_2), val_3, x)
        return x + self.loc

    def _sample_n(self, shape, seed):
        # `shape` is the requested *sample* shape.
        # We need to combine it with the *batch* shape.
        full_shape = _tf.concat([shape[None], self.batch_shape_tensor()], axis=0)

        # We use `stateless_normal` which is the modern TFP/JAX way.
        p = _tf.random.stateless_uniform(
            shape=full_shape, seed=seed, dtype=self.dtype)

        samples = self._quantile(p)
        return samples

    def _mean(self):
        # The mean is just `loc`. We must ensure it broadcasts to the
        # full batch_shape.
        return _tf.broadcast_to(self.loc, self.batch_shape_tensor())

    def _variance(self):
        e = _tf.broadcast_to(self.epsilon, self.batch_shape_tensor())
        c = _tf.broadcast_to(self.scale, self.batch_shape_tensor())

        n = 3 * c * e * (c * e + 2) + 6 + (c * e) ** 3
        d = 3 * c ** 2 * (c * e + 1)
        return n / d


class Huber(EpsilonInsensitive):
    """
    Based on the Huber loss.
    """

    def __init__(self,
                 loc,
                 scale,
                 epsilon,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="Huber"):
        """
        Initialize the distribution.

        Args:
          loc: The mean (mu) of the distribution.
          scale: The rate (c) of the distribution.
          epsilon: The insensitivity parameter.
        """
        super(Huber, self).__init__(
            loc, scale, epsilon,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)

    def _log_prob(self, value):
        # Convert value to the correct dtype, just in case.
        value = _tf.convert_to_tensor(value, dtype=self.dtype)
        thr = self.epsilon #/ self.scale

        z = _tf.math.abs(value - self.loc) / self.scale

        log_density = _tf.where(
            _tf.less_equal(z, thr),
            - 0.5 * z ** 2,
            - thr * (z - 0.5 * thr)
        )

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6))
        log_density = log_density - _tf.math.log(norm)

        return log_density

    def _cdf(self, value):
        std = self.scale
        thr = self.epsilon #/ std

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6))

        x = (value - self.loc) / std

        val_1 = _tf.math.exp(thr * x + 0.5 * thr ** 2) / (thr + 1e-6)
        val_2 = _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6) \
                + _np.sqrt(_np.pi / 2) * (_tf.math.erf(thr / _np.sqrt(2))
                                          + _tf.math.erf(x / _np.sqrt(2)))
        val_3 = _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6) \
                + 2 * _np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2)) \
                + (_tf.math.exp(-0.5 * thr ** 2) -
                   _tf.math.exp(0.5 * thr ** 2 - thr * x)) / (thr + 1e-6)

        prob = _tf.where(_tf.greater(x, -thr), val_2, val_1)
        prob = _tf.where(_tf.greater(x, thr), val_3, prob)

        prob = prob / norm
        return prob

    def _quantile(self, p):
        std = self.scale
        thr = self.epsilon #/ std

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6))

        p_1 = _tf.math.exp(- 0.5 * thr ** 2) / (thr + 1e-6) / norm
        # p_2 = p_1 + _np.sqrt(_np.pi) * _tf.math.erf(thr) / norm
        p_2 = p_1 + 2 * _np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2)) / norm

        val_1 = (_tf.math.log(norm * thr * p) - 0.5 * thr**2) / (thr + 1e-6)
        val_2 = (norm * p - _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6)) / _np.sqrt(_np.pi / 2) \
                - _tf.math.erf(thr / _np.sqrt(2))
        val_2 = _np.sqrt(2) * _tf.math.erfinv(val_2)
        val_3 = - (_tf.math.log(norm * thr * (1 - p)) - 0.5 * thr**2) / (thr + 1e-6)
        # val_3 = - (_tf.math.log(- (p - p_2) * norm * thr + _tf.math.exp(-0.5 * thr ** 2)) - 0.5 * thr ** 2) / (thr + 1e-6)

        x = _tf.where(_tf.greater(p, p_1), val_2, val_1)
        x = _tf.where(_tf.greater(p, p_2), val_3, x)
        return x * std + self.loc

    def _variance(self):
        std = self.scale
        thr = self.epsilon #/ std

        norm = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                    + _tf.math.exp(-0.5 * thr ** 2) / (thr + 1e-6))

        n1 = 2 * (_np.sqrt(_np.pi / 2) * _tf.math.erf(thr / _np.sqrt(2))
                  - thr * _tf.math.exp(-0.5 * thr ** 2))
        n2 = 2 * _tf.math.exp(-0.5 * thr ** 2) \
             * (thr + 2 / (thr + 1e-6) + 2 / (thr + 1e-6) ** 3)
        return (n1 + n2) / norm * std ** 2


def hazen_plotting_positions(n, dtype=_tf.float64):
    """
    Generates plotting positions using the Hazen formula.
    p_i = (i - 0.5) / n for i = 1, ..., n
    """
    n = _tf.cast(n, dtype=dtype)
    i = _tf.range(1., n + 1., dtype=dtype)
    return (i - 0.5) / n


class SplineBased(_tfd.Distribution):
    """
    A continuous distribution based on a sample ECDF.

    The distribution is defined by:
    - A MonotonicSpline for the body (between min/max samples)
    - Exponential tails for extrapolation (below min / above max)

    The tails are 'stitched' to the spline by matching the
    value and slope at the min/max sample points.
    """

    def __init__(self,
                 x_sorted,
                 prob,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="SplineBased"):
        parameters = dict(locals())
        with _tf.name_scope(name) as name:
            # --- 1. Setup Data and Spline ---
            dtype = _tf.float64
            self._x_sorted = x_sorted

            self._x_min = self._x_sorted[:1]
            self._x_max = self._x_sorted[-1:]

            # Get plotting positions (our p_i values)
            self._p_i = prob
            self._p_min = self._p_i[:1]
            self._p_max = self._p_i[-1:]

            # Create the spline for the main body
            # self.spline = _gint.MonotonicCubicSpline()
            self.spline_cdf = _gint.StatefulMonotonicCubicSpline(self._x_sorted, self._p_i)
            self.spline_quantile = _gint.StatefulMonotonicCubicSpline(self._p_i, self._x_sorted)

            # --- 2. Solve for Exponential Tails ---

            # Get slopes at the join points
            s_min = self.spline_cdf.interpolate(self._x_min, grad=True)
            s_max = self.spline_cdf.interpolate(self._x_max, grad=True)

            # Solve lower tail: F(x) = c_L * exp(lambda_L * x)
            self._lambda_lower = s_min / self._p_min
            self._c_lower = self._p_min * _tf.exp(-self._lambda_lower * self._x_min)

            # Solve upper tail: 1 - F(x) = c_U * exp(-lambda_U * x)
            self._lambda_upper = s_max / (1.0 - self._p_max)
            self._c_upper = (1.0 - self._p_max) * _tf.exp(self._lambda_upper * self._x_max)

            super().__init__(
                dtype=dtype,
                reparameterization_type=_tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    # @_tf.function
    def _cdf(self, x):
        """Calculates the Cumulative Distribution Function (CDF)."""
        # Calculate the value for all three regions
        cdf_lower = self._c_lower * _tf.exp(self._lambda_lower * x)
        cdf_body = self.spline_cdf.interpolate(x)
        cdf_upper = 1.0 - self._c_upper * _tf.exp(-self._lambda_upper * x)

        # Use _tf.where to stitch them together
        # If x > x_max, use upper.
        # Else, if x < x_min, use lower.
        # Else, use body.
        cdf = _tf.where(x > self._x_max, cdf_upper,
                        _tf.where(x < self._x_min, cdf_lower, cdf_body))
        return cdf

    def _log_cdf(self, x):
        return _tf.math.log(self._cdf(x))

    # @_tf.function
    def _prob(self, x):
        """Calculates the Probability Density Function (PDF)."""
        # PDF is the derivative of the CDF
        pdf_lower = self._c_lower * self._lambda_lower * _tf.exp(self._lambda_lower * x)
        pdf_body = self.spline_cdf.interpolate(x, grad=True)
        pdf_upper = self._c_upper * self._lambda_upper * _tf.exp(-self._lambda_upper * x)

        # Stitch them together
        pdf = _tf.where(x > self._x_max, pdf_upper,
                        _tf.where(x < self._x_min, pdf_lower, pdf_body))

        # Ensure pdf is non-negative (spline slope might be ~0)
        return _tf.maximum(pdf, 0.0)

    def _log_prob(self, x):
        # We add a small epsilon to avoid log(0)
        prob = self._prob(x)
        return _tf.math.log(prob + 1e-38)

    # @_tf.function
    def _quantile(self, p):
        """Calculates the Quantile Function (Inverse CDF)."""
        # Inverse functions for the tails
        q_lower = (_tf.math.log(p) - _tf.math.log(self._c_lower)) / self._lambda_lower
        q_upper = (_tf.math.log(self._c_upper) - _tf.math.log(1.0 - p)) / self._lambda_upper

        # Inverse function for the body (the spline)
        q_body = self.spline_quantile.interpolate(p)

        # Stitch them together
        q = _tf.where(p > self._p_max, q_upper,
                      _tf.where(p < self._p_min, q_lower, q_body))
        return q


class SmoothEmpirical(SplineBased):
    """
    A continuous distribution based on a sample ECDF.

    The distribution is defined by:
    - A MonotonicSpline for the body (between min/max samples)
    - Exponential tails for extrapolation (below min / above max)

    The tails are 'stitched' to the spline by matching the
    value and slope at the min/max sample points.
    """
    def __init__(self,
                 samples,
                 name="SmoothEmpirical"):
        with _tf.name_scope(name) as name:
            dtype = _tf.float64
            samples = _tf.convert_to_tensor(samples, dtype=dtype)
            x_sorted = _tf.sort(samples, axis=0)

            # Get plotting positions (our p_i values)
            n_batch = _tf.shape(x_sorted)[1]
            n_samples = _tf.cast(_tf.shape(x_sorted)[0], dtype=dtype)
            prob = _tf.tile(hazen_plotting_positions(n_samples, dtype=dtype)[:, None], [1, n_batch])

            super().__init__(x_sorted, prob, name=name)


@_tf.function
def hazen_binned_plotting_points(samples, num_bins, margin=0.05, pseudo_counts=1, dtype=_tf.float64):
    """
    Calculates K plotting points based on binned data.

    Args:
      samples: A 1D tensor of raw data.
      num_bins: The number of bins (K) to create.
      margin: The margin to use.
      pseudo_counts: The number of pseudo counts to use.
      dtype: The data type to use.

    Returns:
      (x_prime, p_prime): A tuple of K x-coordinates and K p-coordinates.

    """
    # 1. Get total sample size
    n = _tf.cast(_tf.shape(samples)[0], dtype=dtype) + pseudo_counts

    # 2. Define bin edges
    data_min = _tf.reduce_min(samples)
    data_max = _tf.reduce_max(samples)
    dif = data_max - data_min
    data_min, data_max = data_min - dif * margin / 2, data_max + dif * margin / 2
    bin_edges = _tf.linspace(data_min, data_max, num_bins + 1)

    # 3. Get histogram counts (c_j)
    counts = _tf.histogram_fixed_width(samples, [data_min, data_max], nbins=num_bins)
    counts = _tf.cast(counts, dtype=dtype) + _tf.cast(pseudo_counts / num_bins, dtype=dtype)

    # 4. Calculate x_prime (bin centers)
    bin_width = (data_max - data_min) / _tf.cast(num_bins, dtype=dtype)
    # The x-coordinate for each bin is its center
    x_prime = bin_edges[:-1] + bin_width / 2.0

    # 5. Calculate p_prime (Hazen-style cumulative frequency)
    # This implements: p'_j = (1/n) * (sum(c_1..c_{j-1}) + c_j/2)

    # Probability of each bin (f_j = c_j / n)
    f_j = counts / n

    # Cumulative probability *before* bin j (P_{j-1})
    p_j_minus_1 = _tf.cumsum(f_j, exclusive=True)

    # p'_j = P_{j-1} + f_j / 2
    p_prime = p_j_minus_1 + f_j / 2.0

    return x_prime, p_prime


class BinnedEmpirical(SplineBased):
    def __init__(self,
                 samples,
                 num_bins,
                 margin=0.05,
                 pseudo_counts=1,
                 name="BinnedEmpirical"):
        with _tf.name_scope(name) as name:
            dtype = _tf.float64
            num_bins = _tf.constant(num_bins, _tf.int32)

            x_binned, probs = [], []
            for s in _tf.unstack(samples, axis=1):
                xb, p = hazen_binned_plotting_points(
                    s, num_bins=num_bins, margin=margin, pseudo_counts=pseudo_counts, dtype=dtype
                )
                x_binned.append(xb)
                probs.append(p)
            x_binned = _tf.stack(x_binned, axis=1)
            probs = _tf.stack(probs, axis=1)

            super().__init__(x_binned, probs, name=name)


class EmpiricalGaussianMixture:
    def __init__(self, samples, num_knots, batch_size=1000, epochs=None):
        self.num_knots = num_knots
        self.batch_size = batch_size
        n_samples = _tf.shape(samples)[0]
        n_data = _tf.shape(samples)[1]

        if epochs is None:
            batches_per_epoch = _np.maximum(n_data // batch_size, 1)
            epochs = _np.maximum(500 // batches_per_epoch, 100)

        self.x_binned = _tf.sort(samples, axis=0)  # [n_samples, batch]
        self.probs = hazen_plotting_positions(n_samples)[:, None]

        self.x_min = _tf.reduce_min(self.x_binned, axis=0, keepdims=True)
        self.x_max = _tf.reduce_max(self.x_binned, axis=0, keepdims=True)
        x_norm = (self.x_binned - self.x_min) / (self.x_max - self.x_min)  # [n_samples, n_data]
        x_norm = _tf.transpose(x_norm)[:, :, None]  # [n_data, n_samples, 1]

        # Gaussian mixture weights
        self.logits = _tf.Variable(_tf.zeros([n_data, num_knots]), _tf.float64)  # [batch, bins]
        self.knots = _tf.linspace(_tf.constant(0.0), 1.0, num_knots)
        self.base_dist = _tfd.Normal(loc=self.knots[None, None, :], scale=1.0 / self.num_knots)

        # training
        dataset = _tf.data.Dataset.range(_tf.cast(n_data, _tf.int64)).batch(self.batch_size)
        optimizer = _tf.keras.optimizers.Adam(learning_rate=0.01)

        @_tf.function
        def train_step(indices_chunk):
            with _tf.GradientTape() as tape:
                mat = self.base_dist.cdf(_tf.gather(x_norm, indices_chunk))  # [batch, n_samples, num_bins]
                c_raw_chunk = _tf.gather(self.logits, indices_chunk)
                c_batch = _tf.nn.softmax(c_raw_chunk, axis=1)  # [batch, num_bins]
                y_hat_batch = _tf.einsum('cab,cb->ac', mat, c_batch)
                loss = _tf.reduce_mean(_tf.square(y_hat_batch - self.probs))

            # Calculate gradients and apply them to our one variable
            gradients = tape.gradient(loss, [self.logits])
            optimizer.apply_gradients(zip(gradients, [self.logits]))

            return loss

        print(f"Fitting {n_data} distributions in chunks of {batch_size}...")
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for indices_chunk in dataset:
                loss = train_step(indices_chunk)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

        print("Training finished.")

    def cdf(self, x, batch_size=1000):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)   # [new_knots, batch]
        x_norm = _tf.transpose(x_norm)[:, :, None]

        dataset = _tftools.batched_dataset(x_norm, batch_size, shuffle=False)
        probs = []
        for (batch, idx) in dataset:
            mat = self.base_dist.cdf(batch)  # [batch, new_knots, old_knots]
            c_raw_chunk = _tf.gather(self.logits, idx)  # [batch, old_knots]
            c_batch = _tf.nn.softmax(c_raw_chunk, axis=1)
            p = _tf.einsum('cab,cb->ac', mat, c_batch)
            probs.append(p)

        return _tf.concat(probs, axis=1)

    def prob(self, x, batch_size=1000):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)  # [new_knots, batch]
        x_norm = _tf.transpose(x_norm)[:, :, None]

        dataset = _tftools.batched_dataset(x_norm, batch_size, shuffle=False)
        probs = []
        for (batch, idx) in dataset:
            mat = self.base_dist.prob(batch)  # [batch, new_knots, old_knots]
            c_raw_chunk = _tf.gather(self.logits, idx)  # [batch, old_knots]
            c_batch = _tf.nn.softmax(c_raw_chunk, axis=1)
            p = _tf.einsum('cab,cb->ac', mat, c_batch)
            probs.append(p)

        return _tf.concat(probs, axis=1)

    def quantile(self, p, batch_size=1000):
        # p is [n_knots, batch]
        amp = self.x_max - self.x_min
        low = _tf.reduce_min(self.x_min - 10 * amp)
        high = _tf.reduce_max(self.x_max + 10 * amp)

        dataset = _tftools.batched_dataset(_tf.transpose(p), batch_size, shuffle=False)

        quants = []
        for batch, _ in dataset:
            results = _tfp.math.find_root_chandrupatla(
                lambda z: self.cdf(z) - _tf.transpose(batch),
                low=low,
                high=high,
                max_iterations=100,
            )
            quants.append(results[0])

        return _tf.concat(quants, axis=1)
