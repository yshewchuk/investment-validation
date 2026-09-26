"""Byte-identical v2 copies of legacy panel/feature math, plus a new
per-decision-date daily-market-state extraction core.

Ported from ``engine/data/features/panel.py`` and ``engine/features.py``.
Legacy keeps its own copy of this math unchanged; this module does not
import from legacy and legacy does not import from this module — two
independent copies of the same math, not a shared import, per the project's
"pure functions MOVE, legacy stays frozen until cutover" rule. A missing
value is an absent key in the returned mapping, never a fabricated NaN/0.0.
"""
from __future__ import annotations

import bisect
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SPANS",
    "history_features",
    "add_implied_history",
    "daily_state_lookup",
]

SPANS = (2, 4, 8, 12)


def _causal_ema(history: list[float], span: int) -> float | None:
    """EMA over events strictly before the current one.

    Reproduces the same recursion as the legacy panel exactly: seed on the
    first prior event and step forward with ``a = 2/(span+1)``, returning
    None until there are at least ``span`` prior events. ``pandas.ewm`` is
    *not* substitutable here — it differs in seeding and in the
    ``adjust=True`` weighting.
    """
    if len(history) < span:
        return None
    a = 2.0 / (span + 1.0)
    ema = history[0]
    for value in history[1:]:
        ema = a * value + (1.0 - a) * ema
    return ema


def history_features(
    prior_moves: Sequence[float],
    prior_abs: Sequence[float],
) -> dict[str, float | None]:
    """Event-history features for the event that follows the given history.

    ``prior_*`` must contain events strictly before the one being scored.
    Byte-identical copy of ``engine.data.features.panel.history_features``.
    """
    out: dict[str, float | None] = {
        "n_prior": len(prior_moves),
        "mean_prior_move": float(np.mean(prior_moves)) if len(prior_moves) else None,
        "mean_prior_abs_move": float(np.mean(prior_abs)) if len(prior_abs) else None,
    }
    for span in SPANS:
        out[f"ema{span}_prior_move"] = _causal_ema(list(prior_moves), span)
        out[f"ema{span}_prior_abs_move"] = _causal_ema(list(prior_abs), span)
    return out


def _anchor_index(
    series_dates: np.ndarray,
    event_dates: np.ndarray,
    as_of_dates: np.ndarray | None,
) -> np.ndarray:
    """Row index each market block is read at: the earlier of two ceilings.

    Byte-identical copy of ``engine.data.features.panel._anchor_index``.
    Strictly before the event date, and on-or-before any given ``as_of``
    date; taking the ``min`` composes them. Callers MUST check for a
    negative (out-of-range) result before indexing with it — this function
    returns ``-1`` for an event before the first series date, it does not
    clamp to ``0``.
    """
    event_idx = np.searchsorted(series_dates, event_dates, side="left") - 1
    if as_of_dates is None:
        return event_idx
    as_of_idx = np.searchsorted(series_dates, as_of_dates, side="right") - 1
    return np.minimum(event_idx, as_of_idx)


def add_implied_history(df: pd.DataFrame) -> pd.DataFrame:
    """``mean_prior_or_implied`` — the running mean of prior quoted implied moves.

    Byte-identical copy of ``engine.data.features.panel.add_implied_history``.
    Strictly prior, expanding, per ticker. Requires an ``or_implied`` column
    already resident on ``df`` (this function does no I/O of its own).
    """
    out = df.sort_values(["ticker", "date"]).copy()
    prior = out.groupby("ticker")["or_implied"].shift(1)
    out["mean_prior_or_implied"] = (
        prior.groupby(out["ticker"]).expanding().mean().reset_index(level=0, drop=True)
    )
    return out


DAILY_STATE_FIELDS: Mapping[str, str] = {
    "implied_move": "im",
    "iv10": "iv10",
    "iv30": "iv30",
    "exern_iv10": "exern_iv10",
    "exern_iv30": "exern_iv30",
    "iee": "iee",
    "skew": "skew",
    "contango": "contango",
    "fwd90_30": "fwd90_30",
    "fexern90_30": "fexern90_30",
    "rvol30": "rvol30",
    "spot": "spot",
    "mcap_log": "mcap_log",
}

DAILY_STATE_LAGS: tuple[int, ...] = (1, 5, 10)

LAGGED_FIELDS: tuple[str, ...] = ("implied_move", "iv10", "iv30", "exern_iv30")


def _is_present(value: object) -> bool:
    """True unless ``value`` is None or a NaN float (numpy or built-in)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def daily_state_lookup(
    rows: Sequence[Mapping[str, object]],
    decision_date: object,
) -> dict[str, float]:
    """Market state for one ticker at one decision date, on-or-before rule.

    ``rows`` are that ticker's own daily rows (already filtered to one
    ticker by the caller), each row a mapping with at least a ``"date"`` key
    (any type ``decision_date`` is directly comparable to, e.g.
    ``pandas.Timestamp`` or ``datetime.date``), a ``"src_iv"`` key, and the
    keys of ``DAILY_STATE_FIELDS`` (``"implied_move"``, ``"iv10"``, ...).

    Reads every value at the last row on-or-before ``decision_date`` — not
    strictly-before — because ``decision_date`` is a close at which we would
    trade, and that close's own quotes are known then. This factors the
    per-date rule out of ``engine.features.daily_state_frame`` (which does
    the same thing across every row of a request DataFrame at once); this
    function does it for a single date, with no pandas dependency.

    Only rows carrying an IV surface (``"src_iv"`` present) are eligible, to
    match legacy excluding market-cap-only rows from being the as-of answer.

    A missing value — no eligible row at or before ``decision_date``, or a
    lag whose reference row does not exist, or a source field that is itself
    absent/NaN on the chosen row — is an ABSENT KEY in the returned mapping,
    never ``float("nan")`` and never a fabricated ``0.0``.
    """
    surface = [row for row in rows if _is_present(row.get("src_iv"))]
    if not surface:
        return {}
    surface = sorted(surface, key=lambda row: row["date"])
    dates = [row["date"] for row in surface]
    # side="right" - 1 == "the last row on or before decision_date", matching
    # engine.features.daily_state_frame's own searchsorted rule exactly.
    idx = bisect.bisect_right(dates, decision_date) - 1
    if idx < 0:
        return {}

    out: dict[str, float] = {}
    current = surface[idx]
    for source_field, output_key in DAILY_STATE_FIELDS.items():
        value = current.get(source_field)
        if _is_present(value):
            out[output_key] = float(value)

    for source_field in LAGGED_FIELDS:
        output_key = DAILY_STATE_FIELDS[source_field]
        current_value = current.get(source_field)
        if not _is_present(current_value):
            continue
        for lag in DAILY_STATE_LAGS:
            prior_idx = idx - lag
            if prior_idx < 0:
                continue
            prior_value = surface[prior_idx].get(source_field)
            if not _is_present(prior_value):
                continue
            out[f"{output_key}_d{lag}"] = float(current_value) - float(prior_value)

    return out
