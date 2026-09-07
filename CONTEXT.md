# geoML

Machine learning models for spatial and geoscientific data, built on
variational Gaussian processes. This file is the project's glossary: what
each word means here, and which near-synonyms to avoid. It carries no
implementation detail — that is what `CLAUDE.md`, the reference pages and
the design records under `docs/` are for.

## What is being modelled

**Ground**:
The quantity that exists at a location, independent of anyone measuring it.
What a prediction reports, with the measurement error already averaged out.
_Avoid_: truth, reality, actual value

**Measurement**:
A reading somebody took, which scatters around the ground by the
measurement error. What an assay is, and what a model must be compared
against — never against the ground it predicts.
_Avoid_: observation, sample (a sample is a piece of rock, not a number)

**Support**:
The volume a value stands for: a core length, a block, a point. Two numbers
of the same variable are only comparable on the same support.
_Avoid_: scale, resolution

**Realization**:
One draw of a whole field, internally consistent across locations, so that
a quantity computed from several locations at once is honest. A set of them
is an *ensemble*.
_Avoid_: simulation run, sample path, scenario

**Cut-off**:
The value a decision turns on — a mining cut-off, a contaminant limit.
Declared on a variable and carried to whatever is predicted from it.
_Avoid_: threshold, grade limit

**Unit**:
What a variable's values are measured in — percent, ppm, g/t. A label on a
variable the model reads directly; also the divisor on a part of a
composition, whose parts must reach a common whole before they can be added
up. Not to be confused with **extent**, which is what a grade-tonnage curve
accumulates: a length, an area, a volume, or a mass.
_Avoid_: dimension, scale

## The three variances

The package distinguishes three questions that are all called "uncertainty"
elsewhere. Keeping them apart is the point.

**Latent variance**:
How sure the model is of the ground at a location.
_Avoid_: model error, kriging variance

**Dispersion**:
How much the ground varies *inside* a block. A well-known block can still be
heterogeneous, and that is what decides whether cutting it finer would tell
anyone anything.
_Avoid_: within-block variance, internal variability

**Noise variance**:
How far a fresh *measurement* of a location would fall from its ground
value. What has to be added back before comparing a prediction with an
assay.
_Avoid_: nugget (a nugget is one model of it, not the quantity)

## Containers and what they hold

**Container**:
A set of locations plus everything known or predicted at them — points, a
grid, a block model, a mesh. Addressed by **path**, never by attribute
chain: `container.get("assay/Zn/prediction")`.
_Avoid_: dataset, dataframe, object

**Variable**:
One modelled quantity on a container: continuous, vector, compositional,
categorical, binary. What a model trains on and predicts into.
_Avoid_: field, attribute, property

**Component**:
One part of a vector or compositional variable, addressed one level down —
`Elements/Zn`. A composition's components share a whole; a vector's do not.
_Avoid_: element, part (except of a composition), column

**Category**:
One class of a categorical variable, addressed like a component but
answering with probabilities and indicators rather than values.
_Avoid_: class, label (see below), domain

**Metadata**:
A per-location fact the models never see — the hole a sample came from, its
depth, its length, a filter naming ground worth predicting. It describes the
sample or the place, not the quantity being modelled.
_Avoid_: attribute, auxiliary variable

**Block**:
One cell of a block model, carrying its own origin and size. Values on it
are on block support, and a **sub-block** is a position inside it used to
work out what the block's value should be — never a thing predicted in its
own right.
_Avoid_: cell, voxel, panel

## The model

**Latent network**:
The composed graph of latent-variable nodes that produces the field a
likelihood observes. The modelling structure, as distinct from the
likelihood that connects it to data.
_Avoid_: architecture, layers

**Inducing point**:
One of the locations the variational approximation is pinned at. They are
chosen, not trained, and their number is what governs cost.
_Avoid_: pseudo-input, knot, support point

**Expert**:
One subset of inducing points covering part of the domain, so that a large
model is many small ones. **Computational tiling, not a statistical
regime** — experts overlap deliberately, and no expert means anything
geologically.
_Avoid_: cluster, partition, local model

**Variational state**:
The parameters that encode the *data* in a trained model, as opposed to the
hyperparameters that encode the structure. What cross-validation
re-initializes so that a fold begins ignorant of what it must predict.
_Avoid_: weights, posterior

**Warping**:
The transformation between the latent Gaussian field and a variable's own
scale. A **chain** of them is applied in order, and whether one *mixes* —
whether one component of its input can reach another of its output —
decides how the likelihood integrates the noise.
_Avoid_: link function, transform (a transform acts on coordinates here)

**Transform**:
A map applied to *coordinates* before a kernel sees them — anisotropy, a
projection, a fault's displacement. The spatial counterpart of a warping,
and never a synonym for it.
_Avoid_: warp, mapping

**Likelihood**:
What connects a latent field to measured values, holding the measurement
error and the back-transformation. It holds no data of its own.
_Avoid_: loss, observation model

## Validation

**Fold**:
One group of locations held out together, chosen so that predicting it
resembles the real prediction task rather than being flattered by
neighbours.
_Avoid_: split, partition

**Out-of-fold**:
Predicted by a model that never saw the location. The only honest basis for
a score.
_Avoid_: test, held-out prediction

**Declustering**:
Weighting locations so that a statistic describes the *field* rather than
the sampling, which is denser where the answer was already interesting.
_Avoid_: debiasing, weighting
