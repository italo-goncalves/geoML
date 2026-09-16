# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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
"""What a long call is doing, and how to stop it.

`geoml.progress(callback)` is a context manager: every long call made inside
it reports what it has finished, and the callback raising is how the caller
cancels. The callback travels on a `contextvars.ContextVar` rather than an
argument, because the long calls nest -- `refine` predicts, `cross_validate`
trains and predicts, a mesh set contours -- and threading an argument down
each of those paths would put it in a dozen signatures to serve one purpose.

A cancelled call leaves its object consistent and says nothing else: the
exception the callback raised travels out untouched, so the caller can tell
a cancelled run from a finished one. What "consistent" means per task is in
`progress`'s docstring, and it is the contract the tests hold the sites to.
"""
import contextlib as _contextlib
import contextvars as _contextvars
import dataclasses as _dataclasses

from collections.abc import Callable as _Callable

__all__ = ["Cancelled", "Progress", "progress"]


_CALLBACK: "_contextvars.ContextVar[_Callable[[Progress], None] | None]" = (
    _contextvars.ContextVar("geoml_progress_callback", default=None))
_WITHIN: "_contextvars.ContextVar[tuple[str, ...]]" = (
    _contextvars.ContextVar("geoml_progress_within", default=()))


class Cancelled(Exception):
    """Raised by a progress callback to stop the call reporting to it.

    geoML never raises this itself, and never catches it: it is a name for
    the caller to raise so that the cancel reads as one at the other end.
    Any exception cancels -- this one only says why.
    """


@_dataclasses.dataclass(frozen=True)
class Progress:
    """One report from a call running inside :func:`geoml.progress`.

    Attributes
    ----------
    task
        What is running: `"train"`, `"predict"`, `"refine"`,
        `"cross_validate"` or `"mesh_set"`.
    done
        Units finished, counted in whatever `unit` names. Never decreases
        within one task.
    total
        Units expected, or `None` where the count is not known in advance.
        A cap rather than a promise wherever training may stop early.
    unit
        What `done` counts: `"iteration"`, `"epoch"`, `"batch"`, `"pass"`,
        `"fold"`, `"body"` or `"realization"`.
    bound
        The evidence lower bound, on the tasks that have one, else `None`.
    within
        The enclosing tasks, outermost first -- `("refine",)` for the
        predictions a refinement makes, `()` at the top.
    """
    task: str
    done: int
    total: "int | None" = None
    unit: str = "step"
    bound: "float | None" = None
    within: "tuple[str, ...]" = ()


@_contextlib.contextmanager
def progress(callback: "_Callable[[Progress], None] | None"):
    """Report what long calls are doing, and cancel them by raising.

    Every call made inside the block reports through `callback`, which
    receives one :class:`Progress` per unit finished. Raising from the
    callback cancels: the exception travels out of the geoML call
    untouched, and what was finished stays finished.

    Parameters
    ----------
    callback
        Called with one :class:`Progress` argument. `None` disables
        reporting for the block, which is how a caller silences an inner
        call inside an outer reporting one.

    Notes
    -----
    What a cancelled call leaves behind, per task:

    - `train`: the model at the last completed iteration or batch, its
      parameters refreshed and its `training_log` appended to, saveable and
      trainable again from there.
    - `predict`: the batches that finished, written into the target;
      `unpredicted()` names the locations left, and predicting only those
      gives what predicting the lot would have.
    - `refine`: nothing. Each pass builds a new block model and only the
      return hands it over, so a cancel loses what the passes made. Write
      the loop out instead -- predict, `needs_splitting`, `split` -- which
      is the documented way to stop part way and keep the model.
    - `cross_validate`: nothing -- the scores are of the whole, so a
      cancelled run has none to give.
    - `mesh_set`: the bodies and realizations already contoured, recorded
      in the store, which opens and says it is incomplete. A realization
      still being contoured when the cancel comes is discarded with its
      worker; only what was recorded survives.

    Reporting is per *finished* unit, so `done` is what would survive a
    cancel at that moment rather than what is being attempted.

    Examples
    --------
    .. code-block:: python

        def watch(event):
            print(event.task, event.done, "of", event.total)
            if stop_file.exists():
                raise geoml.Cancelled()

        with geoml.progress(watch):
            model.train_full(max_iter=2000)
            model.predict(blocks, n_sim=50)
    """
    token = _CALLBACK.set(callback)
    try:
        yield
    finally:
        _CALLBACK.reset(token)


@_contextlib.contextmanager
def reporting(task, total=None, unit="step"):
    """Marks a task, and hands back the function its body reports through.

    `report(done, bound=None)` emits one event. The task's own name is not
    in its events' `within`: that names what encloses it, which is what a
    caller needs to tell a refinement's predictions from a bare one.

    Yields a function that does nothing at all when nobody is listening, so
    a site pays one `ContextVar.get` per unit and no more.
    """
    within = _WITHIN.get()
    token = _WITHIN.set(within + (task,))

    def report(done, bound=None, total=total):
        callback = _CALLBACK.get()
        if callback is not None:
            callback(Progress(task=task, done=done, total=total, unit=unit,
                              bound=bound, within=within))

    try:
        yield report
    finally:
        _WITHIN.reset(token)


def emit(task, done, total=None, unit="step", bound=None):
    """Reports one unit without marking a task, for a site that cannot.

    `reporting` sets a `ContextVar` for the length of its block, which a
    *generator* must not do: a generator body runs in its caller's context,
    so the mark would leak out at every `yield` and outlive an abandoned
    one. `_over_batches` is such a generator, and it needs no mark anyway --
    nothing reports from inside a prediction batch, so no event would ever
    carry `"predict"` in its `within`.
    """
    callback = _CALLBACK.get()
    if callback is not None:
        callback(Progress(task=task, done=done, total=total, unit=unit,
                          bound=bound, within=_WITHIN.get()))


def listening():
    """Whether anything is listening, for a site whose report costs work."""
    return _CALLBACK.get() is not None
