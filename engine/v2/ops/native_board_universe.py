"""Enumerate the native-covered board universe: one request per event per
native-covered strategy, plus one DYN-SV meta-request per event.

Mirrors `engine.score.score_calendar`'s event × strategy enumeration (the ATM
pass only) for the strategies native has an input builder for, without
constructing a legacy `Scorer` or loading the legacy chain index. This module
never imports `engine.score` — reading `engine.score.DISABLED_STRATEGIES`
would pull in `engine.replay` -> `engine.fills` at import time, which the
isolation invariant forbids even when nothing in that chain is ever called.

`DYN-SV` is a meta-row, not a tenth covered strategy: one `BoardRequest` per
event, never one per strategy — it fans out internally to the seven
dynamic-menu members at scoring time, the same shape legacy uses when it
appends its chooser row once per event after scoring the frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.registry.strategies import DYNAMIC_MENU
from engine.v2.scoring.source_inputs import SUPPORTED_STRATEGIES

__all__ = ["BoardRequest", "board_requests"]

#: Every event gets exactly this strategy set (sorted for determinism, never
#: a frozenset's own iteration order — that order is not stable across a
#: process restart with a different PYTHONHASHSEED), plus one DYN-SV request.
#:
#: This module deliberately does not import `engine.strategy_policy` (or
#: `engine.score`/`engine.structures`) even for a read-only consistency
#: check against `DISABLED_STRATEGIES`/`STRUCTURES`: any v2 -> legacy import
#: must be declared in checks/legacy_adapters.json, whose adapter count may
#: only shrink, and engine.v2.ops already has its one allowed adapter module
#: (`legacy_adapter.py`, §4.2). `SUPPORTED_STRATEGIES` already excludes both
#: disabled strategies by construction (it comes from native's own input
#: builder, which has no entry for CAL-P/CND-P), so that check would only
#: restate a fact this module's one real dependency already guarantees.
_COVERED_STRATEGIES = tuple(sorted(SUPPORTED_STRATEGIES))
_REQUIRED_COLUMNS = ("ticker", "event_date", "session")

# Defensive consistency check, paid once at import time — a v2-only
# frozenset operation over names already imported above, no I/O, no legacy
# dependency:
assert frozenset(DYNAMIC_MENU) <= SUPPORTED_STRATEGIES, (
    "every dynamic-menu member must be native-covered"
)


@dataclass(frozen=True, slots=True)
class BoardRequest:
    """A pure board-enumeration key: no fill model, no strike, no snapshot
    binding. Translating this into a full native `ScoreRequest` is later
    work, once a snapshot/deployment/model binding exists to resolve it
    against."""

    ticker: str
    strategy: str
    event_date: pd.Timestamp
    session: str


def board_requests(
    as_of,
    horizon_days: int,
    tickers: Iterable[str] | None,
    events_table: pd.DataFrame,
) -> tuple[BoardRequest, ...]:
    """Every `BoardRequest` the native board scores for the given window.

    Filters exactly like `score_calendar`: `as_of <= event_date <=
    as_of + horizon_days`, `session` not null, and an optional ticker filter.
    Events are ordered by `(event_date, ticker)`. Within one event: the
    native-covered strategies in sorted (alphabetical) order, then one
    `DYN-SV` request last.

    Raises `OpsError` (code `INVALID_REQUEST`) if `events_table` is missing
    any of `ticker`/`event_date`/`session` — a whole-call refusal raised
    before any row is read, never a partial or silently empty result.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in events_table.columns]
    if missing:
        raise OpsError(make_problem(
            "INVALID_REQUEST",
            f"events_table missing required columns: {sorted(missing)}",
        ))

    as_of_ts = pd.Timestamp(as_of).normalize()
    horizon = as_of_ts + pd.Timedelta(days=horizon_days)

    events = events_table[
        (events_table["event_date"] >= as_of_ts)
        & (events_table["event_date"] <= horizon)
        & events_table["session"].notna()
    ]
    if tickers is not None:
        events = events[events["ticker"].isin(set(tickers))]
    events = events.sort_values(["event_date", "ticker"])

    requests: list[BoardRequest] = []
    for event in events.itertuples(index=False):
        ticker = str(event.ticker)
        event_date = pd.Timestamp(event.event_date)
        session = str(event.session)
        for strategy in _COVERED_STRATEGIES:
            requests.append(BoardRequest(ticker, strategy, event_date, session))
        requests.append(BoardRequest(ticker, "DYN-SV", event_date, session))
    return tuple(requests)
