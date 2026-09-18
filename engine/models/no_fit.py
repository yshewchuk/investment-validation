"""Process-wide guard forbidding legacy fitting, training and cache writes.

Phase 5's P5-2 negative control (``guides/rearchitecture_phase5_models.md``):
"Rig all fitting/provider/cache-write paths to fail." :func:`no_fit_guard` is
the one switch that rigs every legacy path reachable from a score request:

- ``engine/data/features/tier4.py::fit_fold`` — the one place a Tier-4 model
  is ever fit.
- ``engine/data/features/tier4.py::serving_model``'s cache-miss branch — it
  calls ``fit_fold`` and then writes the serving cache with ``joblib.dump``.
- ``engine/models/registry.py::ModelArtifact.save`` — ``joblib.dump`` into
  ``data/models``.
- ``engine/models/training/{gate,gate_forecast_analog,implied_t1,
  runup_move,size_model,iv_crush}.py::fit`` — each module's own sklearn
  ``.fit(...)`` entry point (``train()``/``walk_forward``/``fit_final`` in
  ``engine/models/training/common.py`` all reach a model only through this
  one function per module).

This lives under legacy ``engine/`` rather than ``engine/v2/models`` on
purpose: ``checks/import_layers.py`` rule 3 ("legacy never imports v2")
forbids every path above from importing anything under ``engine.v2``.
``engine/v2/models/adapters.py`` keeps its own ``RuntimeFitForbidden``, raised
by ``ReadOnlyArtifact`` when something calls ``.fit()``/``.set_params()`` on
an artifact object already loaded through :class:`engine.v2.models.loader.
FrozenInference` — a different mechanism (attribute interception on one
loaded object) guarding a different moment (after load, during inference).
This module's guard is a standing flag legacy call sites check for themselves
before they would otherwise fit or write a cache file at all.

Thread-local so a test asserting the guard on one thread is never flipped by
a concurrent thread's fitting.
"""
from __future__ import annotations

import contextlib
import threading

__all__ = ["RuntimeFitForbidden", "fitting_forbidden", "forbid_fitting", "no_fit_guard"]

_state = threading.local()


class RuntimeFitForbidden(RuntimeError):
    """A legacy fitting/training/cache-write path ran while the guard was active."""


def fitting_forbidden() -> bool:
    """True while a :func:`no_fit_guard` block is active on this thread."""
    return getattr(_state, "forbidden", False)


def forbid_fitting(path: str) -> None:
    """Raise :class:`RuntimeFitForbidden` if the guard is active.

    Call this as the FIRST statement of every fitting/training/model-cache-
    write path, before any real work. ``path`` names the call site so a test
    (or an error log) can tell which path tripped without guessing.
    """
    if fitting_forbidden():
        raise RuntimeFitForbidden(f"no-fit guard active: {path}")


@contextlib.contextmanager
def no_fit_guard():
    """Forbid every fitting/training/model-cache-write path for the block.

    Re-entrant and thread-local: leaving a nested block restores whatever the
    outer block had set, rather than assuming the outer block wanted it off.
    """
    previous = fitting_forbidden()
    _state.forbidden = True
    try:
        yield
    finally:
        _state.forbidden = previous
