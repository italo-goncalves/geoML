# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
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
"""
The numbers behind the figures.

Nothing here draws anything, and nothing here imports matplotlib. A figure is
one way of looking at these arrays; a dashboard is another, and it reads the
same functions rather than working the values out a second time. It also means
the arithmetic -- which components a fraction of variance asks for, what a
composition looks like once it is opened up -- can be tested against numbers
instead of against pictures.
"""
import numpy as _np
import scipy.stats as _stats

import geoml.data as _data
import geoml.storage as _storage


def variable(container, name):
    """The variable called `name`, or a message saying what there is."""
    try:
        return container.variables[name]
    except KeyError:
        raise KeyError(
            "no variable named %r; found %s"
            % (name, ", ".join(sorted(container.variables)) or "none"))


def variable_or_component(container, name):
    """
    The variable called `name`, or the component of a vector variable.

    A composition is held as one variable, so its parts are not in
    `container.variables` and asking for `"Zn"` would otherwise mean reaching
    into `Elements` by hand.
    """
    if name in container.variables:
        return container.variables[name]

    for var in container.variables.values():
        if isinstance(var, _data.VectorVariable) and name in var.components:
            return var.components[name]

    names = set(container.variables)
    for var in container.variables.values():
        if isinstance(var, _data.VectorVariable):
            names.update(str(label) for label in var.components)
    raise KeyError("nothing named %r here; found %s"
                   % (name, ", ".join(sorted(names)) or "nothing"))


def numeric_values(var):
    """
    A continuous or vector variable as a matrix, and the rows that hold one.

    Returns
    -------
    values : array
        `(n_data, n_columns)`, whatever the variable's length.
    measured : array
        Boolean, `(n_data,)`. A vector variable is measured at a location only
        where every component is.
    labels : list
        A name per column.
    """
    values, has_value = var.get_measurements()
    measured = _np.all(has_value == 1.0, axis=1)
    labels = [str(label) for label in getattr(var, "labels", [var.name])]
    return _np.asarray(values, dtype=float), measured, labels


def category_values(var):
    """
    A categorical variable as one label per location.

    Returns
    -------
    values : array
        The label at each location; the empty string where nothing was
        measured, which is how a missing code reads.
    measured : array
        Boolean, `(n_data,)`.
    labels : list
        The categories actually present, in the variable's own order.
    """
    column = getattr(var, "measurements_a", None)
    if column is None:
        column = getattr(var, "measurements", None)
    if column is None:
        raise TypeError(
            "%s holds no measurements to group by" % type(var).__name__)

    values = _np.asarray(column.to_numpy(), dtype=object)
    measured = values != ""
    present = set(values[measured])
    labels = [str(label) for label in var.labels if label in present]
    return values, measured, labels


def groups(values, measured, labels):
    """A boolean mask per category, in the order the labels come in."""
    return [(label, measured & (values == label)) for label in labels]


def centred_log_ratio(values):
    """
    A composition opened up into ordinary numbers.

    Proportions carry a constant sum, so their covariance is negative by
    construction and a PCA of them describes the constraint as much as the
    data. The log-ratios have no such constraint. This is `warping`'s
    `CenteredLogRatio` in NumPy, for exploring rather than for modelling.
    """
    logged = _np.log(values)
    return logged - _np.mean(logged, axis=1, keepdims=True)


def logarithm(values, compositional=False):
    """
    The data on a log scale, by the road that suits it.

    A composition takes the centred log-ratio: its parts carry a constant sum,
    and logging them one at a time leaves that constraint in place, whereas
    dividing each by the row's geometric mean first removes it. Anything else
    takes an ordinary logarithm.

    Non-positive values stop this rather than quietly becoming infinities:
    which columns they are in is worth knowing, since a zero in an assay
    usually means below detection rather than absent, and what to put in its
    place is a decision about the data, not about the figure.
    """
    values = _np.asarray(values, dtype=float)

    if _np.any(values <= 0):
        bad = _np.where(_np.any(values <= 0, axis=0))[0]
        raise ValueError(
            "a logarithm needs positive values; columns %s hold zeros or "
            "negatives. Replace them first -- half the smallest positive "
            "value is the usual choice for a detection limit"
            % ", ".join(str(int(column)) for column in bad))

    return centred_log_ratio(values) if compositional else _np.log(values)


def principal_components(values, explained=0.9):
    """
    The principal components of `values`, down to a share of the variance.

    Follows `warping.PCA`: an eigendecomposition of the covariance, taken
    largest first. The scores are the plain projection, which is what a biplot
    puts on its axes; the loadings are the eigenvectors, for the arrows.

    Parameters
    ----------
    values : array
        `(n_data, n_columns)`.
    explained : float
        The share of the total variance to reach. The number of components is
        the fewest that reach it -- never fewer than two, so that there is a
        plot to draw, and never more than there are columns.

    Returns
    -------
    dict
        `scores` `(n_data, n_components)`, `loadings`
        `(n_columns, n_components)`, `ratio` (the share each component carries,
        all of them), `n_components`, and -- for drawing the components back
        onto the data they came from -- `mean`, the centre they turn about,
        and `eigenvalues`, the variance along each, whose square root is a
        length in the data's own units.
    """
    values = _np.asarray(values, dtype=float)
    mean = _np.mean(values, axis=0, keepdims=True)
    centred = values - mean
    covariance = centred.T @ centred / centred.shape[0]

    eigenvalues, eigenvectors = _np.linalg.eigh(covariance)
    order = _np.argsort(eigenvalues)[::-1]
    eigenvalues = _np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    total = _np.sum(eigenvalues)
    ratio = eigenvalues / total if total > 0 else _np.zeros_like(eigenvalues)

    reached = int(_np.searchsorted(_np.cumsum(ratio), explained) + 1)
    n_components = int(_np.clip(reached, 2, values.shape[1]))

    return {"scores": centred @ eigenvectors[:, :n_components],
            "loadings": eigenvectors[:, :n_components],
            "ratio": ratio,
            "n_components": n_components,
            "mean": mean.ravel(),
            "eigenvalues": eigenvalues}


def component_analysis(var, explained=0.9, log=False):
    """
    `principal_components` for a variable, on a log scale if asked.

    A composition is opened up whether or not `log` is set, and that is not an
    oversight: its parts carry a constant sum, so their covariance is singular
    and the first component of the raw proportions would spend itself
    describing the closure rather than the data. There is no useful PCA of a
    composition to refuse it. Everything else is left as measured unless `log`
    says otherwise.
    """
    values, measured, labels = numeric_values(var)
    values = values[measured]

    compositional = isinstance(var, _data.CompositionalVariable)
    if compositional or log:
        values = logarithm(values, compositional)
        labels = ["%s(%s)" % ("clr" if compositional else "log", label)
                  for label in labels]

    analysis = principal_components(values, explained)
    analysis["labels"] = labels
    analysis["measured"] = measured
    return analysis


def warped_values(model, name):
    """
    The measurements as the model sees them, after its warping.

    A model does not work on the data as measured: the likelihood's warping
    takes it somewhere it can be treated as Gaussian and, for several
    variables at once, uncorrelated. Whether it succeeded is a thing to look
    at rather than assume, and looking at it means passing the data through
    the same warping the model trained with -- already fitted, since
    `VGPNetwork` initializes each likelihood from the measurements.

    Only measured rows go through. `get_measurements` fills the rest with 1.0
    to keep the array rectangular, and a warping has no reason to be kind to a
    value that was never there.

    Returns
    -------
    values : array
        `(n_measured, warping.size_out)`.
    measured : array
        Boolean, `(n_data,)`, saying which rows those are.
    labels : list
        A name per column.
    """
    names = list(model.variables)
    if name not in names:
        raise KeyError("the model does not hold %r; it holds %s"
                       % (name, ", ".join(names)))

    likelihood = model.likelihoods[names.index(name)]
    warping = getattr(likelihood, "warping", None)
    if warping is None:
        raise TypeError(
            "%s carries no warping, so there is nothing to transform"
            % type(likelihood).__name__)

    values, measured, _ = numeric_values(variable(model.data, name))
    warped, _ = warping.forward(values[measured])
    warped = _np.asarray(warped, dtype=float)

    # The columns are numbered rather than named after what was measured. A
    # warping may rotate the data, or bend it, or take seven columns to five,
    # and then a column is a mixture of the measurements rather than one of
    # them -- calling it Zn would be wrong in a way nobody would catch.
    labels = ["Variable %d" % (i + 1) for i in range(warped.shape[1])]
    return warped, measured, labels


def _strided(store, stride):
    """Every `stride`-th value of a store, flattened, read a band at a time.

    The same values as `asarray(store).reshape(-1)[::stride]`, arrived at
    without the materialization. Each band picks up where the last one left
    off, so which values are taken depends on the stride alone and not on where
    the chunk boundaries happen to fall -- two stores of the same shape are
    thinned to the same positions, which is what lets the columns be paired.
    """
    n_columns = store.shape[1] if len(store.shape) > 1 else 1
    pieces = []
    for band in store.row_bands():
        values = _np.asarray(store[band], dtype=float).reshape(-1)
        pieces.append(values[(-band.start * n_columns) % stride::stride])

    return _np.concatenate(pieces) if len(pieces) > 0 else _np.zeros(0)


def simulation_sample(var, most=100000):
    """
    The simulated values, thinned to a number that can be drawn.

    A simulated variable holds `n_data x n_sim` values per component, which on
    a block model runs to hundreds of millions -- more than a figure can show
    and, spread across every pair of a matrix, more than memory should be
    asked to hold. What a distribution looks like is settled by far fewer, so
    a stride is taken through the block -- and taken while reading, a band of
    locations at a time, so the values that are thrown away are never all in
    memory together either.

    The stride is the same for every component, and the mask that drops
    non-finite values is applied to whole rows. Thinning the components
    separately would pair the value simulated at one location with the value
    simulated at another, which is not a pair the model ever produced -- the
    joint shape, which is the thing being looked at, would be an artefact.

    Returns
    -------
    values : array
        `(n_sample, n_columns)`, columns in the variable's own order.
    """
    components = getattr(var, "components", None)
    parts = [var] if components is None \
        else [components[label] for label in var.labels]

    stride, columns = None, []
    for part in parts:
        # asked for simulations it does not have, a variable hands back
        # `asarray(None)` rather than raising, which would go on to fail
        # somewhere less informative
        store = getattr(part, "simulations", None)
        if store is None:
            raise ValueError(
                "%r carries no simulations to compare against; predict with "
                "n_sim greater than zero first" % var.name)

        if stride is None:
            stride = max(1, int(_np.prod(store.shape)) // int(most))
        columns.append(_strided(store, stride))

    values = _np.column_stack(columns)
    return values[_np.all(_np.isfinite(values), axis=1)]


def padded_range(values, margin=0.1):
    """
    The span of each column, opened up by `margin` at both ends.

    Returns a `(low, high)` pair per column. A column of one repeated value
    has no span to open up, so it is given a unit of room rather than a window
    of zero width.
    """
    values = _np.asarray(values, dtype=float)
    low = _np.min(values, axis=0)
    high = _np.max(values, axis=0)
    room = _np.where(high > low, (high - low) * margin, 1.0)
    return list(zip(low - room, high + room))


def color_limits(values, clip=None):
    """
    The ends of a colour scale, set by where the data mostly is.

    One value far from the rest takes the whole of a colour scale with it and
    leaves everything else in a single shade -- the usual fate of a
    geochemical assay, whose tail is long and whose interest is not in it.
    Naming a pair of quantiles ends the scale where the data mostly ends
    instead: `[0, 0.99]` for a variable skewed to the right, `[0.01, 0.99]` to
    take both tails.

    Nothing is dropped and nothing is altered. The values keep their own
    numbers and a hover still reports what was measured; what is bounded is
    the scale, so the few beyond it take the end colour rather than setting
    it.

    Returns `(low, high)`, or None when `clip` is None -- and None again when
    the two ends come out equal, since a scale of zero width is no scale.
    """
    if clip is None:
        return None

    clip = _np.asarray(clip, dtype=float).ravel()
    if len(clip) != 2 or not _np.all((clip >= 0) & (clip <= 1)) \
            or clip[0] >= clip[1]:
        raise ValueError(
            "clip takes two quantiles between 0 and 1, the lower first: "
            "[0, 0.99] keeps the long right tail off the scale, [0.01, 0.99] "
            "takes both ends. Got %r" % (clip.tolist(),))

    values = _np.asarray(values, dtype=float).ravel()
    values = values[_np.isfinite(values)]
    if len(values) == 0:
        return None

    low, high = _np.quantile(values, clip)
    return None if low >= high else (float(low), float(high))


def color_choices(container, names):
    """
    Several variables' values over the locations that carry all of them.

    For a figure that keeps one set of points and swaps the values over them.
    The locations are those measured in *every* one of the variables named,
    which is the point: a cloud that gained and lost points as the choice
    changed would be two things changing at once, and the comparison -- this
    variable against that one, here -- is exactly what would be lost.

    A name may be a variable, one component of a vector variable, or a vector
    variable itself, which stands for all of its components in order. So
    `["Elements"]` names seven grades and `["Cd", "Zn"]` names two.

    Returns
    -------
    values : array
        `(n_kept, n_columns)`, a column per choice in the order asked for.
    rows : array
        Which locations those are, as indices into the container.
    labels : list
        A name per column.
    """
    columns, labels = [], []
    measured = _np.ones(container.n_data, dtype=bool)

    for name in names:
        var = variable_or_component(container, name)
        if isinstance(var, (_data.RockTypeVariable, _data.BinaryVariable)):
            raise TypeError(
                "%r is categorical, and a menu of choices colours by a scale "
                "rather than by a legend; name continuous variables here and "
                "draw the categorical one on its own" % name)

        values, has_value, names_here = numeric_values(var)
        for i, label in enumerate(names_here):
            columns.append(values[:, i])
            labels.append(label)
        measured &= has_value

    if not _np.any(measured):
        raise ValueError(
            "no location carries all of %s at once, and a menu holds one set "
            "of points for every choice on it" % ", ".join(labels))

    rows = _np.where(measured)[0]
    return _np.column_stack(columns)[rows], rows, labels


def cells(n_points, most):
    """
    How many cells to count `n_points` into, at most `most` of them.

    The two halves of a comparison rarely hold comparable numbers -- a few
    hundred measurements against a hundred thousand simulated values -- and
    binning both the same way leaves the sparse half as scattered single counts
    that read as noise. Roughly the square root of the count keeps several
    points in a typical cell either way.
    """
    return int(_np.clip(_np.sqrt(n_points), 10, most))


def counts_2d(x, y, bins=60, log=False):
    """
    Points counted into cells, with the empty ones left empty.

    A cell holding nothing comes back as NaN rather than as zero, so that
    whatever draws it can leave it unpainted: painted, it takes the bottom of
    the colour scale and fills the panel with a background that looks like
    data.

    Returns
    -------
    dict
        `x` and `y`, the cell centres; `z`, what the colour is to be taken
        from, which is the count or its base-ten logarithm; and `count`, the
        count itself, so that a hover can say how many points a cell holds
        whichever of the two is being coloured. `z` and `count` are shaped
        `(n_y, n_x)`, the way an image is indexed.
    """
    counted, x_edges, y_edges = _np.histogram2d(
        _np.asarray(x, dtype=float), _np.asarray(y, dtype=float), bins=bins)

    # a heatmap is indexed by row, and a row runs along y
    counted = counted.T
    empty = counted == 0
    values = _np.where(empty, _np.nan, counted)

    return {"x": 0.5 * (x_edges[:-1] + x_edges[1:]),
            "y": 0.5 * (y_edges[:-1] + y_edges[1:]),
            "z": _np.log10(values) if log else values,
            "count": counted.astype(int)}


def density_grid(x, y, grid=60, most=4000):
    """
    Smoothed density over a pair of columns, on a regular mesh.

    The estimate costs one pass over the data for every mesh cell, so a long
    column is thinned first: a density is a shape, and a few thousand points
    settle it as well as a few million do.

    Returns `(x_axis, y_axis, density)` with `density` shaped
    `(grid, grid)` -- rows along `y_axis`, as an image is -- or None when
    there is nothing to smooth: fewer than three points, or a column that
    never varies, which has no width for a kernel to sit in.
    """
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    if len(x) > most:
        step = len(x) // most
        x, y = x[::step], y[::step]
    if len(x) < 3 or _np.ptp(x) == 0 or _np.ptp(y) == 0:
        return None

    x_axis = _np.linspace(_np.min(x), _np.max(x), grid)
    y_axis = _np.linspace(_np.min(y), _np.max(y), grid)
    mesh_x, mesh_y = _np.meshgrid(x_axis, y_axis)

    kernel = _stats.gaussian_kde(_np.vstack([x, y]))
    density = kernel(_np.vstack([mesh_x.ravel(), mesh_y.ravel()]))
    return x_axis, y_axis, density.reshape(grid, grid)


def density_curve(values, limits, points=200, most=5000):
    """
    A smoothed distribution as a line, across `limits`.

    Thinned as `density_grid` is, and for the same reason. Returns
    `(grid, density)`, or None when there is nothing to smooth.
    """
    values = _np.asarray(values, dtype=float)
    if len(values) > most:
        values = values[::len(values) // most]
    if len(values) < 3 or _np.ptp(values) == 0:
        return None

    grid = _np.linspace(limits[0], limits[1], points)
    return grid, _stats.gaussian_kde(values)(grid)


def normal_curve(values, low, high, points=200):
    """
    The normal of the same mean and spread as `values`, across a window.

    Fitted rather than standard, so that the curve asks about the shape alone.
    A warping's parameters are trained along with everything else and need not
    leave the data at unit variance -- the GP's amplitude absorbs a scale
    factor -- so a standard normal would call a perfectly symmetric result
    skewed. Returns `(x, density)`, or None for a column that never varies.
    """
    values = _np.asarray(values, dtype=float)
    mean, deviation = _np.mean(values), _np.std(values)
    if deviation <= 0:
        return None

    x = _np.linspace(low, high, points)
    density = _np.exp(-0.5 * ((x - mean) / deviation) ** 2) \
        / (deviation * _np.sqrt(2 * _np.pi))
    return x, density


def prediction_values(container, name):
    """
    What was measured against what was predicted, component by component.

    Returns
    -------
    measured_values, predicted_values : array
        `(n_compared, n_columns)`, holding only the locations that carry both.
    labels : list
        A name per column.
    rows : array
        Which locations those are, as indices into the container. A figure
        that can be brushed needs to say which row each of its points came
        from, and only the rows carrying both a measurement and a prediction
        are drawn.
    """
    var = variable(container, name)
    components = getattr(var, "components", None)
    parts = [var] if components is None else \
        [components[label] for label in var.labels]

    measured_values, predicted_values, labels = [], [], []
    for part in parts:
        if getattr(part, "prediction", None) is None:
            raise ValueError(
                "%r carries no prediction; run the model over this data first"
                % name)
        measured_values.append(part.measurements.values.to_numpy())
        predicted_values.append(part.prediction.values.to_numpy())
        labels.append(str(part.name))

    measured_values = _np.stack(measured_values, axis=1).astype(float)
    predicted_values = _np.stack(predicted_values, axis=1).astype(float)

    both = _np.all(~_np.isnan(measured_values), axis=1) \
        & _np.all(~_np.isnan(predicted_values), axis=1)
    if not _np.any(both):
        # a variable is built with a prediction column already in place, empty
        # until a model fills it, so this is what never having run one looks
        # like -- as well as what a genuine mismatch looks like
        raise ValueError(
            "%r has no location carrying both a measurement and a prediction; "
            "run the model over this data first" % name)

    return (measured_values[both], predicted_values[both], labels,
            _np.where(both)[0])


def moving_average(values, window):
    """
    The running mean of `values`, and where each point belongs.

    Returns the positions as well: a mean over `window` points only exists
    once there are that many, so the curve starts later than the one it
    smooths and has to be drawn against its own x.
    """
    values = _np.asarray(values, dtype=float)
    window = int(window)
    if window <= 1 or window > len(values):
        return _np.arange(1, len(values) + 1), values.copy()

    kernel = _np.ones(window) / window
    smoothed = _np.convolve(values, kernel, mode="valid")
    return _np.arange(window, len(values) + 1), smoothed


def training_curve(model, window=None):
    """
    The training log, with a running mean over it.

    Each entry is one optimizer step -- and under `train_svi` one *batch*, so
    the curve is noisy by construction: every value is the ELBO estimated from
    a sample of the data and a sample of the latent variables. The mean is what
    says whether it is still climbing.

    Parameters
    ----------
    window : int
        Points to average over. Defaults to a fiftieth of the log, which keeps
        the smoothing proportionate to however long training ran, but never
        fewer than five: a mean of two or three smooths nothing, and draws a
        second line on top of the first saying the same thing. A window longer
        than the log leaves it alone, and then there is no second line at all.
    """
    values = _np.asarray(getattr(model, "training_log", []), dtype=float)
    if len(values) == 0:
        raise ValueError("this model has not been trained yet")

    if window is None:
        window = int(max(5, len(values) // 50))
    position, smoothed = moving_average(values, window)

    return {"iteration": _np.arange(1, len(values) + 1), "value": values,
            "smooth_iteration": position, "smooth": smoothed,
            "window": window}


def realizations(var):
    """
    A variable's simulations, or its prediction as the only one there is.

    Returns `(n_data, n_realizations)` either way, so that whatever reads it
    does not have to care which it got.
    """
    values = None
    try:
        values = _np.asarray(var.get_simulations(), dtype=float)
    except Exception:
        # a vector variable with nothing simulated fails stacking its
        # components rather than handing anything back
        pass

    # and a continuous one hands back `asarray(None)`, a dimensionless nan,
    # which reads as a value right up until it is asked for its length
    if values is None or values.ndim == 0:
        prediction = getattr(var, "prediction", None)
        if prediction is None:
            raise ValueError(
                "%r carries neither simulations nor a prediction" % var.name)
        values = _np.asarray(prediction.values.to_numpy(), dtype=float)

    return values.reshape(len(values), -1)


def realization_store(var):
    """
    A variable's realizations as a store to read in bands, not as an array.

    The same `(n_data, n_realizations)` that `realizations` hands back, without
    asking memory for all of it at once. A block model's simulations are the
    one thing a container holds that will not fit -- hundreds of gigabytes is
    an ordinary size for them -- and every use of them here is a reduction over
    locations, which never needs more than a band of rows at a time.

    A variable with no simulations falls back on its prediction, which is a
    single column and already in RAM, and comes back as a one-band store so
    that the caller has only the one path to write.
    """
    store = getattr(var, "simulations", None)
    if store is not None and len(getattr(store, "shape", ())) == 2:
        return store
    return _storage.ArrayStore.from_numpy(realizations(var))


def block_density(container, density, n_data, n_realizations):
    """
    A density per block, to turn a volume into a tonnage.

    `density` may be a number, the name of a metadata column, or the name of a
    `ContinuousVariable`. Only the last of these can be uncertain, and when it
    is, its realizations are matched one to one with the grade's: simulation
    `i` of the density belongs with simulation `i` of the grade, and pairing
    them any other way would invent a correlation that was never modelled.

    Comes back as a single number where the density is one, and otherwise as a
    store to read in bands alongside the grade. A simulated density is exactly
    as big as the grade, so materializing it would give back everything not
    materializing the grade saves.
    """
    if density is None:
        return 1.0

    if isinstance(density, (int, float)):
        return float(density)

    if density in container.metadata:
        values = _np.asarray(container.get_metadata(density), dtype=float)
        return _storage.ArrayStore.from_numpy(values.reshape(n_data, 1))

    if density in container.variables:
        store = realization_store(variable(container, density))
        if store.shape[1] not in (1, n_realizations):
            raise ValueError(
                "%r carries %d realizations and the grade carries %d; they "
                "have to be matched one to one, or the density has to be the "
                "same in all of them"
                % (density, store.shape[1], n_realizations))
        return store

    raise KeyError(
        "nothing named %r to take a density from; metadata holds %s and the "
        "variables are %s"
        % (density, ", ".join(sorted(container.metadata)) or "nothing",
           ", ".join(sorted(container.variables))))


def containing_variable(container, var):
    """The variable `var` is a component of, if it is a component of one."""
    for candidate in container.variables.values():
        components = getattr(candidate, "components", None) or {}
        if any(component is var for component in components.values()):
            return candidate
    return None


def _column_of(obj, name):
    """
    An attribute of `obj` that is a column of values, or None.

    A column that was never filled does not count as found. A vector
    variable's components are ordinary continuous variables, so each carries a
    `latent_variance` whether or not a model ever wrote one; stopping at the
    empty one would hide the column on the parent that does hold something,
    and then every block would fail the filter for being NaN.
    """
    attribute = getattr(obj, name, None)
    if attribute is None or not hasattr(attribute, "values"):
        return None

    values = _np.asarray(attribute.values.to_numpy(), dtype=float)
    return None if _np.all(_np.isnan(values)) else values


def uncertainty_values(container, name, variables=()):
    """
    A number per location saying how much the model doubts itself there.

    Which number that is depends on what was modelled, so it is named rather
    than guessed at. `name` may be:

    - **an array**, one value per location. Whatever the number is and wherever
      it came from, it can always be handed over directly, which is the way out
      of every case the names below do not reach.
    - **`"Variable.column"`**, naming exactly where to read it --
      `"Elements.uncertainty"` while the grade is one of its components, or a
      column belonging to some other variable entirely.
    - **a bare name**, looked for on each of `variables` in turn and then among
      the metadata columns.

    `variables` is what a bare name is tried against: the grade, and then the
    variable containing it. That second one is not a nicety. A component of a
    vector variable has `latent_variance` set to None and no uncertainty of its
    own -- the column that exists belongs to the parent, so grading `Zn` and
    asking for `"uncertainty"` has to reach `Elements` to find anything.
    """
    if not isinstance(name, str):
        values = _np.asarray(name, dtype=float).ravel()
        if len(values) != container.n_data:
            raise ValueError(
                "an uncertainty given as an array needs one value per "
                "location: got %d for %d" % (len(values), container.n_data))
        return values

    if "." in name:
        owner, _, column = name.partition(".")
        values = _column_of(variable_or_component(container, owner), column)
        if values is None:
            raise KeyError("%r carries no column called %r" % (owner, column))
        return values

    for var in variables:
        values = _column_of(var, name)
        if values is not None:
            return values

    if name in container.metadata:
        return _np.asarray(container.get_metadata(name), dtype=float)

    carried = sorted({key for var in variables for key, value
                      in vars(var).items() if hasattr(value, "values")})
    raise KeyError(
        "nothing named %r to take an uncertainty from. The variables in hand "
        "carry %s; the metadata holds %s. Name a column on another variable "
        "as 'Variable.column', or pass the values themselves"
        % (name, ", ".join(carried) or "no columns",
           ", ".join(sorted(container.metadata)) or "nothing"))


def _grade_band(store, band, keep):
    """One band of realizations, with the blocks that were filtered out gone."""
    values = _np.asarray(store[band], dtype=float)
    return values if keep is None else values[keep[band]]


def _mass_band(density, band, keep, shape):
    """The mass of every block of a band, broadcast to the grade's shape."""
    if not hasattr(density, "shape"):
        return _np.broadcast_to(float(density), shape)

    values = _np.asarray(density[band], dtype=float)
    if keep is not None:
        values = values[keep[band]]
    return _np.broadcast_to(values, shape)


def _cutoff_range(store, bands, keep, name):
    """The span of the finite realizations, in one pass over the store."""
    low, high = _np.inf, -_np.inf
    for band in bands:
        values = _grade_band(store, band, keep)
        finite = values[_np.isfinite(values)]
        if finite.size > 0:
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))

    if not _np.isfinite(low):
        raise ValueError("%r holds no values to cut" % name)
    return low, high


def grade_tonnage(container, name, density=None, cutoffs=30,
                  uncertainty=None, max_uncertainty=None):
    """
    How much material sits above a cut-off, and how good it is.

    Each block contributes its volume, or its mass where a density is given,
    to every cut-off its grade clears. Simulations are carried through
    separately rather than averaged first: the curve of the mean model is not
    the mean of the curves, since a cut-off is a threshold and averaging either
    side of it gives different answers.

    The simulations are read a band of blocks at a time and never held whole:
    a block model runs to hundreds of gigabytes of them, and what comes out is
    one small number per cut-off per realization. Each block is placed at the
    highest cut-off it clears and the curve is the running total from the top
    down, so the cost is one pass over the grade rather than one per cut-off.
    Giving `cutoffs` as values rather than as a count saves the pass that would
    otherwise be needed to find their range.

    Parameters
    ----------
    container
        A gridded container -- the volume of a block comes from its spacing.
    name : str
        The variable to take as the grade.
    density : float or str
        A number, a metadata column, or a `ContinuousVariable`. Without one the
        curve is in volume.
    cutoffs : int or array-like
        The grades to cut at, or how many of them to spread evenly across the
        range of the data.
    uncertainty : str or array
        Where to read how sure the model is at each block: a column name, a
        `"Variable.column"` naming which variable it belongs to, or the values
        themselves. See `uncertainty_values`. A bare name is looked for on the
        grade, then on the variable containing it, then in the metadata.
    max_uncertainty : float
        Blocks doubted more than this are left out altogether -- not counted
        at any cut-off, and not counted towards the grade above one. A block
        the model cannot speak for is not tonnage.

    Returns
    -------
    dict
        `cutoff` `(n_cutoffs,)`; `tonnage`, `grade` and `metal`
        `(n_cutoffs, n_realizations)`; `unit`, which is the extent of a block
        or `"mass"` depending on whether a density was given; and `kept` and
        `total`, the blocks that survived the uncertainty filter and the
        blocks there were.
    """
    step = getattr(container, "step_size", None)
    if step is None:
        raise TypeError(
            "grade-tonnage needs a gridded container, which is what carries "
            "the size of a block; %s does not" % type(container).__name__)
    volume = float(_np.prod(step))
    # a block of a two-dimensional grid has an area, not a volume, and saying
    # otherwise on the axis of a figure someone is reading off is not harmless
    extent = {1: "length", 2: "area", 3: "volume"}.get(len(step), "volume")

    var = variable_or_component(container, name)
    grade = realization_store(var)
    n_data, n_realizations = grade.shape

    mass_per_volume = block_density(
        container, density, n_data, n_realizations)

    total, keep = n_data, None
    if max_uncertainty is not None:
        if uncertainty is None:
            raise ValueError(
                "there is no uncertainty column to filter by; name one when "
                "the Explorer is built, or pass uncertainty=")
        owner = containing_variable(container, var)
        doubt = uncertainty_values(
            container, uncertainty,
            [var] + ([owner] if owner is not None else []))
        keep = _np.isfinite(doubt) & (doubt <= float(max_uncertainty))
        if not _np.any(keep):
            raise ValueError(
                "no block is certain enough to keep: the smallest %r is %g, "
                "above the %g asked for"
                % (uncertainty, _np.nanmin(doubt), max_uncertainty))
    kept = total if keep is None else int(_np.count_nonzero(keep))

    bands = grade.row_bands()
    if isinstance(cutoffs, (int, _np.integer)):
        low, high = _cutoff_range(grade, bands, keep, name)
        cutoffs = _np.linspace(low, high, int(cutoffs))
    cutoffs = _np.asarray(cutoffs, dtype=float)

    # What each band adds is the mass sitting *at* a cut-off -- above it and
    # below the next one up. Summing those from the top down at the end turns
    # them into the mass above each cut-off, which is the curve.
    n_cutoffs = len(cutoffs)
    at_cutoff = _np.zeros([n_cutoffs, n_realizations])
    metal_at_cutoff = _np.zeros_like(at_cutoff)
    columns = _np.arange(n_realizations)

    for band in bands:
        values = _grade_band(grade, band, keep)
        if values.shape[0] == 0:
            continue
        weight = _mass_band(mass_per_volume, band, keep, values.shape)

        # the highest cut-off each block clears; -1 for one that clears none,
        # which is where a block with no value belongs as well
        index = _np.searchsorted(cutoffs, values, side="right") - 1
        index[~_np.isfinite(values)] = -1
        above = index >= 0

        # one bin per (cut-off, realization) pair, so every realization is
        # accumulated in the same pass over the band
        binned = (index * n_realizations + columns)[above]
        values, weight = values[above], weight[above]
        at_cutoff += _np.bincount(
            binned, weights=weight,
            minlength=n_cutoffs * n_realizations
        ).reshape(n_cutoffs, n_realizations)
        metal_at_cutoff += _np.bincount(
            binned, weights=weight * values,
            minlength=n_cutoffs * n_realizations
        ).reshape(n_cutoffs, n_realizations)

    tonnage = volume * _np.cumsum(at_cutoff[::-1], axis=0)[::-1]
    metal = volume * _np.cumsum(metal_at_cutoff[::-1], axis=0)[::-1]

    with _np.errstate(invalid="ignore", divide="ignore"):
        mean_grade = _np.where(tonnage > 0, metal / tonnage, _np.nan)

    return {"cutoff": cutoffs, "tonnage": tonnage, "grade": mean_grade,
            "metal": metal, "unit": extent if density is None else "mass",
            "kept": kept, "total": total}


def grid_shape(n_panels):
    """Rows and columns for `n_panels`, as square as they go."""
    columns = int(_np.ceil(_np.sqrt(n_panels)))
    rows = int(_np.ceil(n_panels / columns))
    return rows, columns
