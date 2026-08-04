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

import geoml.data as _data


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


def simulation_sample(var, most=100000):
    """
    The simulated values, thinned to a number that can be drawn.

    A simulated variable holds `n_data x n_sim` values per component, which on
    a block model runs to hundreds of millions -- more than a figure can show
    and, spread across every pair of a matrix, more than memory should be
    asked to hold. What a distribution looks like is settled by far fewer, so
    a stride is taken through the block. Reading one component at a time keeps
    the high-water mark to a single component's simulations rather than all of
    them at once.

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
        if getattr(part, "simulations", None) is None:
            raise ValueError(
                "%r carries no simulations to compare against; predict with "
                "n_sim greater than zero first" % var.name)

        flat = _np.asarray(part.get_simulations(), dtype=float).reshape(-1)
        if stride is None:
            stride = max(1, len(flat) // int(most))
        columns.append(flat[::stride])

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


def prediction_values(container, name):
    """
    What was measured against what was predicted, component by component.

    Returns
    -------
    measured_values, predicted_values : array
        `(n_compared, n_columns)`, holding only the locations that carry both.
    labels : list
        A name per column.
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

    return measured_values[both], predicted_values[both], labels


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
    try:
        values = _np.asarray(var.get_simulations(), dtype=float)
    except Exception:
        prediction = getattr(var, "prediction", None)
        if prediction is None:
            raise ValueError(
                "%r carries neither simulations nor a prediction" % var.name)
        values = _np.asarray(prediction.values.to_numpy(), dtype=float)

    return values.reshape(len(values), -1)


def block_density(container, density, n_data, n_realizations):
    """
    A density per block, to turn a volume into a tonnage.

    `density` may be a number, the name of a metadata column, or the name of a
    `ContinuousVariable`. Only the last of these can be uncertain, and when it
    is, its realizations are matched one to one with the grade's: simulation
    `i` of the density belongs with simulation `i` of the grade, and pairing
    them any other way would invent a correlation that was never modelled.
    """
    if density is None:
        return _np.ones([n_data, 1])

    if isinstance(density, (int, float)):
        return _np.full([n_data, 1], float(density))

    if density in container.metadata:
        values = _np.asarray(container.get_metadata(density), dtype=float)
        return values.reshape(n_data, 1)

    if density in container.variables:
        values = realizations(variable(container, density))
        if values.shape[1] not in (1, n_realizations):
            raise ValueError(
                "%r carries %d realizations and the grade carries %d; they "
                "have to be matched one to one, or the density has to be the "
                "same in all of them"
                % (density, values.shape[1], n_realizations))
        return values

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


def grade_tonnage(container, name, density=None, cutoffs=30,
                  uncertainty=None, max_uncertainty=None):
    """
    How much material sits above a cut-off, and how good it is.

    Each block contributes its volume, or its mass where a density is given,
    to every cut-off its grade clears. Simulations are carried through
    separately rather than averaged first: the curve of the mean model is not
    the mean of the curves, since a cut-off is a threshold and averaging either
    side of it gives different answers.

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
    grade = realizations(var)
    n_data, n_realizations = grade.shape

    mass = volume * block_density(container, density, n_data, n_realizations)

    total = n_data
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
        grade, mass = grade[keep], mass[keep] if mass.shape[0] > 1 else mass
        if not _np.any(keep):
            raise ValueError(
                "no block is certain enough to keep: the smallest %r is %g, "
                "above the %g asked for"
                % (uncertainty, _np.nanmin(doubt), max_uncertainty))
    kept = grade.shape[0]

    if isinstance(cutoffs, (int, _np.integer)):
        finite = grade[_np.isfinite(grade)]
        if len(finite) == 0:
            raise ValueError("%r holds no values to cut" % name)
        cutoffs = _np.linspace(_np.min(finite), _np.max(finite), int(cutoffs))
    cutoffs = _np.asarray(cutoffs, dtype=float)

    tonnage = _np.zeros([len(cutoffs), n_realizations])
    metal = _np.zeros_like(tonnage)
    for i, cutoff in enumerate(cutoffs):
        above = (grade >= cutoff) & _np.isfinite(grade)
        weight = _np.where(above, mass, 0.0)
        tonnage[i] = _np.sum(weight, axis=0)
        metal[i] = _np.sum(weight * _np.where(above, grade, 0.0), axis=0)

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
