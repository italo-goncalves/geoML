# The catalogue

`geoml.catalogue` (0.7.0) describes every class and function a model or a
script may use, as JSON, for programs that build geoML models without
reading its code. GeoScape's network editor is the first: its networks
round-trip through geoML's saved spec, and it offers nothing the catalogue
has not declared. The format is specified on GeoScape's side, in its
`docs/geoml-catalogue.md`; this record is geoML's half.

## What comes from where

From the code: every key is a class's path as persistence records it
(`module.qualname`, `geoml.latent.network.BasicGP` rather than a re-export),
so a catalogue entry and a saved spec name a class the same way. The
parameters come from `inspect.signature`, which follows `Parametric`'s
wrapper to the real constructor; their types come from the annotations
where the module has them, their descriptions from the docstring's
`Parameters` section, and a class's summary from the first paragraph of its
docstring, or of its constructor's where the class has none. The version,
and the format number a model save carries, are read from the package.

From declarations: every public class of the six catalogued modules
(`latent.network`, `latent.fourier`, `kernels`, `transform`, `warping`,
`likelihood`) carries a `_catalogue` dictionary, assigned in a block where
its module ends: its category, a label, its stability, its parents, how its
size follows from its arguments, whether inducing points pass through it
and whether it needs them, how it chains, the variable types a likelihood
accepts, and the types of arguments no annotation states. The block rather
than the class body keeps the classes as they were and the declarations in
one place per module. `latent/network.py` states every argument's type
there, being outside pyright on purpose; most warping and transform
constructors carry none either.

## Amendments to the format (agreed 2026-09-15)

Four were needed for the declarations to be true, and the spec was revised
with them:

1. `propagates_inducing` may be `"parents"`: true exactly when every parent
   passes inducing points on and all share one root. `Add`,
   `LinearCombination`, `Concatenate` and every one-parent node behave so.
2. `parents` may name the categories a parent must belong to: `GPWalk`'s
   parent must be a GP.
3. A likelihood's size may be `{"rule": "warping"}`: its warping's output
   width, the warping taking the variable's `length`.
4. No default is an object. `BasicInput`, `kernels.Covariance` and
   `kernels.Linear` defaulted to one `Identity()` built at import and
   shared by every instance; they take `transform=None` and build their
   own.

Added while building, and in the spec too: `nullable` on every parameter,
true where `null` is an answer of its own (a prior's strength, where it
switches the prior off). A transform's or warping's `size` is `{"in",
"out"}`, `in` null meaning any width and `same_as_parent` meaning the width
it took; `chain.attaches_to` names the parameter a chain is handed to.

Changed with `GaussianMixture` (0.8.0), the one node whose parents come
in two kinds: `parents` is a list of slots on every node, one per
constructor parameter taking parents and empty for an input -- one form
for a parser rather than an object or a list -- and a slot may carry the
`size` its parent must have (the weights,
one per component: `{"rule": "len", "param": "components"}`); a `common`
size rule may name the slot it reads. And every latent node carries
`gaussian` -- `true`, `false` or `"parents"` -- read off the class
attribute `_GAUSSIAN` rather than declared a second time, because a leaf
that is not Gaussian trains on its realizations.

Added in 0.8.3 for GeoScape's items 35 and 36: `workflow.fit.folds` names
`spatial_k_fold`, and a container's `methods` may hold what its family
declares for itself (`CLASS_METHODS`, inherited by subclasses) -- the block
set's `split`, `crossed_by` and `unbalanced`. Not a shared list by name:
`Mesh3D.split` separates a mesh's pieces and would have been published as
the block set's operation. A mesh argument is typed `data:Mesh3D`.

Added in 0.9.13 for GeoScape's item 47, the format staying 2: `trainable` on
every class that is not internal, empty where it trains nothing, one object
per parameter a program may offer to hold fixed. `name` is what
`parameters[name]` reads, with `{i}` for a family's number and `per` naming
the argument that counts it (`ranges_{i}` per `n_structures`,
`warped_partition_left_{i}` per `size`). `start` is the argument the value
starts from, filled in wherever the parameter has an argument's name, as the
commitment "a trained argument is the parameter of its name" already says;
it is declared where the names differ (`Isotropic`'s `r`) and switched off
where they only coincide (an indicator's `bias`, a flag). `fixed_when` and
`exists_when` are conditions on the constructor call. All a condition's
clauses must hold: `argument` with `equals`, `differs` or `at_least`, and
`parent_size` and `input_dimension`, the two facts no argument says. An input
also has `fixes`, an argument holding fixed everything the object in `param`
trains. The class's own `trainable` is what a program can reach by name,
which is why two things that train are missing from it: the directions a
dynamic anisotropy builds and the components a `Mixture` builds.

Left out by the user's choice: an expert's own state, `alpha_white_i`,
`delta_i` and `bias_i`. The first two are what the data fits and the last is
governed by the node's `bias=`, so nothing in the format has to say "one per
expert" or "variational". Also left out, as never trained: a quadrature's
`nodes`, `Center`'s `std`, `Scale`'s `mean` and `BoxCox`'s found `shift`.

## Found on the way

Each of these would have made a declaration false, and the tests below
caught or would have caught it:

- `Identity()` and `Periodic()` given explicitly could not be saved. Neither
  they nor `_Transform` define `__init__`, `Parametric.__init__` was never
  wrapped, and no arguments were recorded; `__init_subclass__` now wraps
  an inherited initializer nothing has wrapped.
- `Concatenate` declared that it passed inducing points on without asking
  its parents, so a GP on a `Concatenate` of a `Multiply` was built and
  failed at its first refresh. It asks them now, as `Add` does.
- An operation given no parents failed with an `IndexError`, `GPWalk` on
  anything but a GP with an `AttributeError`, and `MultiStructureGP` took
  one structure. All three refuse with a message.
- `warping.__all__` lacked the four parametric links and `kernels.__all__`
  `Covariance` and `RationalQuadratic`.
- `ZScore`'s `std` is declared `float[]`, and a list raised a `TypeError` at
  `std / 100` (found by the test of `start`, 0.9.13).
- The refit `cross_validate` runs freed every `bias_*`, whatever the node's
  `bias=` said, so a box holding one fixed would not have held there (found
  reading item 47's claim that training leaves out what is fixed, 0.9.13).

## Stability

By the user's choice, conservatively: experimental are the two flows, the
four fault transforms, `GaussianInput`, `UncertainInputGP`, `Mixture` and
the legacy `Spline` warping; internal are `geoml.latent.fourier`,
`BellFault2D` (kept for saved models), `NormalizeWithBoundingBox` (a
`BoundingBox` argument no store can hold) and `GradientIndicator` (built by
the model itself for directional data). Everything else is public.

## Every claim, tested

`test_catalogue.py` fails on a public class without a declaration of its
own -- a subclass would otherwise inherit its parent's -- and checks every
declaration against its class: each node built on stand-in parents at two
sizes and its size compared with the rule, a GP placed on it to see whether
inducing points pass, and a parent that passes none put under it; each
transform and warping applied to data of the width it declares; every
likelihood trained a step on every variable type it accepts; every class
that can be offered sent through persistence's encoding and back; what each
class trains against what it registers, every condition built true and false
and every `start` given a value; and the catalogue written by two processes
under different hash seeds, compared byte for byte.

## Left out

`default_warping`, optional in the spec, is not given: what to propose
depends on the variable's sign and closure, which a likelihood class cannot
know. The catalogue is written by `python -m geoml.catalogue` wherever geoML
is installed; publishing it beside each release, so a reader need not
install TensorFlow to have it, would be the docs workflow's job.
