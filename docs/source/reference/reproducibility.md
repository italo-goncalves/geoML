# Reproducibility

What replays, what does not, and the one knob that governs it. A workflow
that records what a run produced needs to know which of its numbers are
promises.

## The one knob

```python
import geoml
geoml.set_seed(1234)      # before anything is built
```

Parameter initialization draws from the package generator that call seeds,
and a model's options draw their own `seed` from it when they are built;
that number governs training's Monte Carlo draws and the simulation stream.
There is no second knob: `GPOptions(seed=...)` is a `TypeError`, and a saved
model keeps the number it drew, so a model reloaded from a store draws what
it drew before.

The call governs what is built **after** it. Building twice after one
`set_seed` gives two different models, which is the point: the generator
moves on.

## What comes back the same

With the same geoML, the same libraries, the same machine and the same
device, and `set_seed` before anything is built:

- the initial parameters, the training log, the predictions, the
  realizations and the measurement samples, bit for bit, in any process;
- a location's realizations whatever the batching, and whichever call
  computed them, given one fit, one seed and one number of realizations --
  so a target predicted in pieces holds the ensemble it would have held
  whole;
- the draws of a node, keyed by its name, which a save replays;
- training split into chunks: it takes the steps one call of the same
  length takes, and stops where it stops;
- a mesh set contoured in worker processes, against one contoured in a
  single process;
- the catalogue `python -m geoml.catalogue` writes.

## What does not

- **Another machine, or another version of anything.** TensorFlow, NumPy
  and the BLAS underneath them choose kernels and summation orders by
  processor and by version, and a different number of threads can reorder a
  reduction. Record the versions beside the numbers.
- **A GPU.** geoML's guarantees above are measured on the CPU. Several
  TensorFlow kernels accumulate in an order that depends on how the device
  schedules the work, so two runs on one machine can differ.
  `tf.config.experimental.enable_op_determinism()` is TensorFlow's own
  switch for that, at a cost in speed this package has not measured. To
  hold a run to the CPU, set `CUDA_VISIBLE_DEVICES=""` in the environment.
- **Another release of geoML.** A fix that changes a number changes it; the
  changelog says which.
- **A helper that takes a `seed` of its own, left at `None`.** Three of
  them cluster, and k-means with no seed starts from NumPy's global state:
  `inducing.from_kmeans`, `inducing.experts` and
  `PointData.spatial_k_fold`. They run before a model exists, so
  `set_seed` has nothing to reach them through -- pass a seed and the
  points, the experts and the folds come back the same.
  `transform.RandomProjections` is the same shape of knob with a fixed
  default, so it replays either way.

## If two runs of one script disagree

In this order: the same versions of geoML and of what it stands on; the same
device; `set_seed` called before the objects are built rather than after;
a seed on every k-means helper the script calls; and nothing reading
NumPy's global generator in between. geoML's own
data-dependent starts are seeded from the package generator -- the robust
covariance behind `RobustPCA` was the last one that was not, and until
0.7.0 it began from one of two fits depending on the process, moving every
number that followed it. `test_seed.py` pins the rule, in this process and
in another.
