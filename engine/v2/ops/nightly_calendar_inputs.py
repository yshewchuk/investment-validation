"""Pinned calendar/expiry/spot staging for cutover PR-6 slice 4a.2.

Composition over three reads, none of which chooses a strike, prices a leg or
touches a provider: the session calendar from the pinned SPY price series
through the decision session, the strategy-eligible listed expiry set from one
exact ``option_chains`` session, and the exact-session pinned raw-close spot.
``generation.resolve_expiry`` applies the native strategy policy to the
candidate days; ``planned_exit_date`` fixes the session-based exit independently
of that expiry; ``nightly_raw_rows.scan_calendar_row`` matches the pinned event
and returns its revision.

Like ``native_board_universe``/``nightly_raw_rows``/``nightly_quote_rows``, this
module never imports ``engine.score``/``engine.structures``/``engine.replay``/
``engine.fills``, and never a legacy calendar or provider call.
"""
from __future__ import annotations

import math
from typing import Any

from engine.v2.contracts import DataQuery, KeyPredicate, PriceQuery, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.price_history_query import get_price_series
from engine.v2.data.repository import Repository
from engine.v2.foundation.market_calendar import (
    CalendarSessions,
    build_calendar_sessions,
    planned_exit_date,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from engine.v2.ops.nightly_raw_rows import CalendarRowInputs, scan_calendar_row
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    validated_as_of,
)

__all__ = ["scan_calendar_row_inputs", "scan_candidate_expiries", "scan_decision_calendar"]

_CALENDAR_TICKER = "SPY"
_CHAIN_TABLE = "option_chains"
_CANDIDATE_COLUMNS = ("right", "strike", "expiry")
#: STR-THRU/STR-RUNUP need a common call-and-put strike; every other native
#: context (put menu, DYN-SV) is a listed put. Mirrors generation's own
#: ``_select_listed_straddle`` (call ∩ put) vs ``_listed_put_expiries`` (put).
_STRADDLE_DOMAIN = frozenset({"STR-THRU", "STR-RUNUP"})
_TRANSLATABLE_REFUSALS = ("NO_EXPIRY_ON_OR_AFTER:", "NO_EXPIRY_DTE_AT_LEAST:")
_NO_RESOLVABLE = "NO_RESOLVABLE_EXPIRY"
_BATCH_CAP = 50_000
_RESULT_CAP = 2_000_000


def _calendar_day(value: Any, field: str) -> str:
    """A naive midnight calendar day, or ``INVALID_REQUEST`` -- never echo the
    submitted value back (ops.errors convention)."""
    try:
        day = validated_as_of(value)
    except NightlySourceBundleRefusal:
        raise fail("INVALID_REQUEST", f"{field} must be a valid naive calendar day") from None
    if day != day.normalize():
        raise fail("INVALID_REQUEST", f"{field} must be a midnight calendar day")
    return day.date().isoformat()


def _require_key(key: Any) -> BoardRequest:
    if not isinstance(key, BoardRequest) or any(
        not isinstance(value, str) or not value.strip()
        for value in (key.ticker, key.strategy, key.session)
    ):
        raise fail("INVALID_REQUEST", "calendar input key requires non-empty identity fields")
    return key


def _date_string(value: Any) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def scan_decision_calendar(repository: Repository, snapshot: SnapshotRef, *,
                           decision_session: Any, event_through: Any) -> CalendarSessions:
    """The pinned session calendar: the SPY price series through the decision
    session is the only source; the observed maximum is its own date, never a
    projected endpoint."""
    session_day = _calendar_day(decision_session, "decision_session")
    through_day = _calendar_day(event_through, "event_through")
    series = get_price_series(repository, PriceQuery(
        ticker=_CALENDAR_TICKER, session_date=session_day, observation_ceiling=session_day,
        lookback_sessions=0), snapshot)
    if not series:
        raise data_fail("CONTRACT_MISMATCH", "no eligible SPY sessions through the decision session")
    source_dates = tuple(sorted({row.date for row in series}))
    return build_calendar_sessions(source_dates, event_through=through_day)


def scan_candidate_expiries(repository: Repository, snapshot: SnapshotRef, key: BoardRequest, *,
                            decision_session: Any) -> tuple[str, ...]:
    """One bounded ``option_chains`` scan of the exact ``(ticker, decision_session)``
    slice, projected to raw ``(right, strike, expiry)``: no other session, no
    quote-usability filter, no expiry before the decision session. The eligible
    domain follows the native strategy policy; the result is the sorted distinct
    ISO tuple, empty only when the domain is empty."""
    _require_key(key)
    session_day = _calendar_day(decision_session, "decision_session")
    contract = repository.table_contract(snapshot, _CHAIN_TABLE)
    version = snapshot.table_versions[_CHAIN_TABLE]
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=_CANDIDATE_COLUMNS,
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(key.ticker,)),
                    KeyPredicate(column="obs_date", operator="eq", values=(session_day,))),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, _BATCH_CAP),
        max_result_rows=min(contract.maximum_result_rows, _RESULT_CAP))
    calls: dict[str, set[float]] = {}
    puts: dict[str, set[float]] = {}
    for batch in repository.scan(query, table_name=_CHAIN_TABLE):
        for row in batch.to_pylist():
            expiry_day = _date_string(row["expiry"])
            if expiry_day < session_day:
                continue
            right = str(row["right"]).upper()
            if right == "C":
                calls.setdefault(expiry_day, set()).add(float(row["strike"]))
            elif right == "P":
                puts.setdefault(expiry_day, set()).add(float(row["strike"]))
    if key.strategy in _STRADDLE_DOMAIN:
        expiries = {day for day in calls if calls[day] & puts.get(day, set())}
    else:
        expiries = set(puts)
    return tuple(sorted(expiries))


def scan_calendar_row_inputs(repository: Repository, snapshot: SnapshotRef, key: BoardRequest, *,
                             decision_session: Any, calendar: CalendarSessions) -> CalendarRowInputs:
    """One row's pinned spot, resolved strategy expiry and independent planned
    exit staged into ``scan_calendar_row``. A missing/unusable exact-session
    spot or a repository failure fails the whole call; an empty eligible expiry
    domain, or the native no-expiry geometry refusals, are the one per-key
    ``NO_RESOLVABLE_EXPIRY`` refusal -- every other geometry/input error
    propagates."""
    from engine.v2.domain.generation import GeometryRefusal, resolve_expiry

    _require_key(key)
    _calendar_day(key.event_date, "event_date")
    session_day = _calendar_day(decision_session, "decision_session")
    series = get_price_series(repository, PriceQuery(
        ticker=key.ticker, session_date=session_day, observation_ceiling=session_day,
        lookback_sessions=0), snapshot)
    spot_row = next((row for row in series if row.date == session_day), None)
    if spot_row is None:
        raise data_fail("CONTRACT_MISMATCH", "no exact-session pinned spot for this ticker")
    spot = spot_row.close_raw
    if spot is None or not math.isfinite(spot) or spot <= 0:
        raise data_fail("CONTRACT_MISMATCH", "exact-session pinned spot close is unusable")
    candidates = scan_candidate_expiries(repository, snapshot, key, decision_session=session_day)
    if not candidates:
        raise _no_resolvable(key)
    inputs = {"event_date": key.event_date, "session": key.session,
              "entry_date": session_day, "quote_date": session_day}
    try:
        expiry = resolve_expiry(key.strategy, inputs, list(candidates))
    except GeometryRefusal as refusal:
        if refusal.code.startswith(_TRANSLATABLE_REFUSALS):
            raise _no_resolvable(key) from refusal
        raise
    exit_date = planned_exit_date(key, calendar)
    return scan_calendar_row(repository, snapshot, key, entry_date=session_day,
                             exit_date=exit_date, expiry=expiry, spot=spot,
                             calendar_observed_through=calendar.observed_through)


def _no_resolvable(key: BoardRequest) -> NativeScoreBatchRowRefusal:
    return NativeScoreBatchRowRefusal(key, _NO_RESOLVABLE, "no strategy-eligible listed expiry")
