"""Build the entry-rule gate's frozen trailing ``pnl_sim`` cutoff (P5-4).

The training-side builder for ``engine.v2.models.trailing_cutoff_artifact``.
Scoring reads the artifact and never calls this module (layer 6 is above
scoring).

The arithmetic is legacy ``engine/pnl_sim.py`` ``trailing_cutoff``, statement
for statement, over ``(event_date, exp_pnl_sim)`` rows the CALLER supplies
(``tools/phase5_datasets.pnl_sim_history``, the stored history file):

* the window is ``[month - window_months, month)`` in calendar months, where
  ``month`` is the first day of the event's month (strictly before it, so an
  event is never ranked against itself or anything later);
* a missing value (``None``/NaN) is dropped (``dropna``); an infinite one is
  kept, as legacy keeps it;
* fewer than ``min_window`` values leave no bar (``None``);
* otherwise the bar is ``np.quantile(values, 1 - quantile)``. A non-finite
  bar (an infinite history value can make one) is frozen as ``None``: the
  rule reads the bar through ``entry_rules._number``, which treats a NaN or
  infinite bar exactly as a missing one.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.models.trailing_cutoff_artifact import (
    PNL_HISTORY_ID,
    TRAILING_MIN_WINDOW,
    TRAILING_QUANTILE,
    TRAILING_WINDOW_MONTHS,
    TrailingCutoffArtifact,
    cutoff_month,
    make_trailing_cutoff_artifact,
)

__all__ = ["build_trailing_cutoff_artifact", "trailing_cutoff_lineage", "window_bounds"]


def window_bounds(month: Any) -> tuple[np.datetime64, np.datetime64]:
    """``(window start, exclusive end)`` in ns for the event month of ``month``."""
    end = np.datetime64(cutoff_month(month)[:7], "M")
    start = end - np.timedelta64(TRAILING_WINDOW_MONTHS, "M")
    return start.astype("datetime64[ns]"), end.astype("datetime64[ns]")


def trailing_cutoff_lineage(month: Any, history_id: str = PNL_HISTORY_ID) -> Lineage:
    """The history rows the bar reads: every event before the month."""
    return Lineage(data=(DataDependency(table=history_id, end_exclusive=cutoff_month(month)),))


def _value(raw: Any) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    return None if value != value else value


def build_trailing_cutoff_artifact(
    rows: Iterable[Mapping[str, Any]],
    *,
    month: Any,
    history_id: str = PNL_HISTORY_ID,
) -> TrailingCutoffArtifact:
    """Freeze the bar for ``month`` from ``{event_date, exp_pnl_sim}`` rows."""
    forbid_fitting("engine.v2.models.training.trailing_cutoff.build_trailing_cutoff_artifact")
    start, end = window_bounds(month)
    prior = []
    for row in rows:
        stamp = np.datetime64(row["event_date"], "ns")
        if np.isnat(stamp) or not start <= stamp < end:
            continue
        value = _value(row.get("exp_pnl_sim"))
        if value is not None:
            prior.append(value)
    cutoff = None
    if len(prior) >= TRAILING_MIN_WINDOW:
        cutoff = float(np.quantile(np.asarray(prior, dtype=float), 1.0 - TRAILING_QUANTILE))
        cutoff = cutoff if np.isfinite(cutoff) else None
    return make_trailing_cutoff_artifact(
        month=month, cutoff=cutoff, lineage=trailing_cutoff_lineage(month, history_id),
        history_id=history_id,
    )
