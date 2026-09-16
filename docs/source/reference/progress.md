# Progress and cancelling

What a long call is doing, and how to stop it. `geoml.progress(callback)` is
a context manager: every long call made inside it reports what it has
finished, and the callback raising is how the caller cancels.

```python
import geoml

def watch(event):
    print(event.task, event.done, "of", event.total)
    if stop_file.exists():
        raise geoml.Cancelled()

with geoml.progress(watch):
    model.train_full(max_iter=2000)
    model.predict(blocks, n_sim=50)
```

The callback travels on a context variable rather than an argument, because
the long calls nest: a refinement predicts, a cross-validation trains and
predicts. An event names the enclosing tasks in `within`, so a refinement's
predictions can be told from a bare one.

`done` counts units *finished*, never units attempted, which is what makes
the number worth acting on: it is what a cancel at that moment would leave
behind.

## Resuming a cancelled prediction

A location's values do not depend on what else is in its batch, so the
batches a cancelled prediction finished are exactly as they would have been.
`unpredicted()` names the rest, on any container:

```python
model.predict(blocks, n_sim=50)                 # cancelled part way
model.predict(blocks, n_sim=50,
              where=blocks.unpredicted())       # finishes the rest
```

The answer is read off the missing values rather than remembered from a
call, so it stays true however the container was arrived at -- reopened from
a store, subsetted, carried into. Each kind of variable declares the column
that marks it: a grade has a `prediction`, a rock type an `entropy`, a
vector variable an `uncertainty`.

A mesh set cancelled part way leaves a store that opens and says it is
incomplete, holding the realizations it finished.

```{eval-rst}
.. autofunction:: geoml.progress

.. autoclass:: geoml.Progress
   :members:

.. autoexception:: geoml.Cancelled
```
