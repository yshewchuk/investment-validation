"""v2-native no-fit guard for the payoff-calibration fitting boundary (P5-4).

``engine/models/no_fit.py`` already guards every LEGACY fitting/training/
cache-write path (P5-2). ``engine/v2`` has no dependency on legacy ``engine/``
code (see ``engine/v2/scoring/native_payoff.py``'s own module docstring), so
the same discipline needs a v2-native copy rather than an import of the
legacy module. This is that copy, scoped today to the one v2 fitting path
that exists: ``engine.v2.scoring.native_payoff.fit_payoff_line`` and
``fit_runup_payoff_surface`` -- the pure re-derivations of
``engine/payoff.py``'s ``np.polyfit``/``np.linalg.lstsq`` calibration.

Reuses :class:`engine.v2.models.adapters.RuntimeFitForbidden` rather than
declaring a second exception type, so a caller that already catches the one
adapters.py raises when a frozen artifact is mutated also catches this one.

Thread-local, exactly like the legacy guard, so a test asserting the guard on
one thread is never flipped by a concurrent thread's fitting.
"""
from __future__ import annotations

import contextlib
import threading

from .adapters import RuntimeFitForbidden

__all__ = ["RuntimeFitForbidden", "fitting_forbidden", "forbid_fitting", "no_fit_guard"]

_state = threading.local()


def fitting_forbidden() -> bool:
    """True while a :func:`no_fit_guard` block is active on this thread."""
    return getattr(_state, "forbidden", False)


def forbid_fitting(path: str) -> None:
    """Raise :class:`RuntimeFitForbidden` if the guard is active.

    Call this as the FIRST statement of every v2 fitting path, before any
    real work. ``path`` names the call site so a test (or an error log) can
    tell which path tripped without guessing.
    """
    if fitting_forbidden():
        raise RuntimeFitForbidden(f"no-fit guard active: {path}")


@contextlib.contextmanager
def no_fit_guard():
    """Forbid every v2 fitting path for the block.

    Re-entrant and thread-local: leaving a nested block restores whatever the
    outer block had set, rather than assuming the outer block wanted it off.
    """
    previous = fitting_forbidden()
    _state.forbidden = True
    try:
        yield
    finally:
        _state.forbidden = previous
