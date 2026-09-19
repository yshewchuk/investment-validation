"""Build the DYN-SV chooser's frozen k-NN analog pool (P5-4).

The training-side builder for ``engine.v2.models.chooser_analog_pool``.
Scoring reads the artifact and never calls this module (layer 6 is above
scoring).

The filter is legacy ``Scorer._chooser_analog_pool`` (``engine/score.py``),
row for row, over rows the CALLER supplies (``tools/build_chooser_pool.py``'s
frame, in its stored order):

* a row with a missing ``exit_date`` or ``pnl`` is dropped (``dropna``: a
  NaN P&L is missing, an infinite one is kept);
* a row without a structure name is dropped (``groupby`` drops NaN keys);
* per structure, a row whose five dimensions are not all finite is dropped;
* the remaining rows keep their given order within the structure.

One rule is added, and it is the causal key: a row that closed on/after
``cutoff`` never enters. Legacy has no cutoff (its file is whatever was last
built); declaring one is what lets a request check which pool it reads.

The artifact stores exit DAYS. Legacy compares full timestamps, and its pool
builder normalizes every exit date to midnight, so the two agree; an exit
timestamp with a time of day is refused here rather than silently truncated.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np

from engine.v2.models.chooser_analog_pool import (
    CHOOSER_ANALOG_DIMS,
    ChooserAnalogPoolArtifact,
    make_chooser_analog_pool_artifact,
)
from engine.v2.models.lineage import Lineage
from engine.v2.models.no_fit import forbid_fitting

__all__ = ["build_chooser_analog_pool_artifact"]


def _exit_day(value: Any) -> str | None:
    """The exit day, ``None`` when missing; refuses a time of day."""
    if value is None or value != value:  # None, NaN or NaT
        return None
    try:
        stamp = np.datetime64(value, "ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unparsable chooser pool exit_date: {value!r}") from exc
    if np.isnat(stamp):
        return None
    day = stamp.astype("datetime64[D]")
    if stamp != day.astype("datetime64[ns]"):
        raise ValueError(f"chooser pool exit_date carries a time of day: {value!r}")
    return str(day)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _kept(row: Mapping[str, Any], bound: str | None) -> tuple[str, tuple] | None:
    strategy = row.get("strategy")
    if strategy is None or strategy != strategy:
        return None
    day = _exit_day(row.get("exit_date"))
    pnl = _number(row.get("pnl"))
    if day is None or pnl != pnl:
        return None
    dims = tuple(_number(row.get(name)) for name in CHOOSER_ANALOG_DIMS)
    if not all(isfinite(value) for value in dims):
        return None
    if bound is not None and day >= bound:
        return None
    return str(strategy), (day, pnl, *dims)


def build_chooser_analog_pool_artifact(
    rows: Iterable[Mapping[str, Any]],
    *,
    pool_id: str,
    cutoff: Any,
    lineage: Lineage,
) -> ChooserAnalogPoolArtifact:
    """Freeze the chooser analog population from ``rows`` in their order.

    ``rows``: ``{strategy, exit_date, pnl, exp_pnl_sim, width_over_forecast,
    n_legs, anchor_over_spot, rel_spread}`` (extra keys ignored).
    """
    forbid_fitting("engine.v2.models.training.chooser_pool.build_chooser_analog_pool_artifact")
    if not isinstance(lineage, Lineage) or not lineage.declared:
        raise ValueError("a frozen chooser analog pool must declare its lineage")
    bound = None if cutoff is None else str(cutoff)[:10]
    grouped: dict[str, list[tuple]] = {}
    for row in rows:
        kept = _kept(row, bound)
        if kept is not None:
            grouped.setdefault(kept[0], []).append(kept[1])
    return make_chooser_analog_pool_artifact(
        pool_id=pool_id, cutoff=bound, strategies=grouped, lineage=lineage,
    )
