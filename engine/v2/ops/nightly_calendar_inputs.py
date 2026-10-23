"""Pinned calendar/expiry/spot staging for the raw-row producer.

Composition over three reads, none of which chooses a strike, prices a leg or
touches a provider: the session calendar from the pinned SPY price series
through the decision session, the strategy-eligible listed expiry set from one
exact ``option_chains`` session (the producer's batched ``chain_rows``, or this module's own per-key scan when none is given), and the exact-session pinned raw-close spot.
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
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from engine.v2.contracts import DataQuery, KeyPredicate, PriceQuery, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.price_history_query import get_price_series
from engine.v2.data.price_history_table import CALENDAR_TICKER
from engine.v2.data.repository import Repository
from engine.v2.foundation.market_calendar import (
    CalendarSessions,
    build_calendar_sessions,
    planned_exit_date,
)
from engine.v2.foundation.market_horizon import CalendarCoverageError, trading_days_before_print
from engine.v2.ops.errors import fail
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch_types import (
    PRICE_HISTORY_NOT_AVAILABLE_DETAIL,
    NativeScoreBatchRowRefusal,
)
from engine.v2.ops.nightly_chain_batch import CHAIN_TABLE, require_chain_rows, require_usable_fields
from engine.v2.ops.nightly_raw_rows import CalendarRowInputs, scan_calendar_row
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    validated_as_of,
)
from engine.v2.scoring.source_inputs import SUPPORTED_STRATEGIES

__all__ = ["scan_calendar_row_inputs", "scan_candidate_expiries", "scan_decision_calendar"]

_CALENDAR_TICKER = CALENDAR_TICKER
_CANDIDATE_COLUMNS = ("right", "strike", "expiry")
#: STR-THRU/STR-RUNUP need a common call-and-put strike; every other native
#: context (put menu, DYN-SV) is a listed put. Mirrors generation's own
#: ``_select_listed_straddle`` (call ∩ put) vs ``_listed_put_expiries`` (put).
_STRADDLE_DOMAIN = frozenset({"STR-THRU", "STR-RUNUP"})
#: The strategy/session domains whose expiry this staging can resolve at all.
#: Strategies: the board universe's own values -- ``native_board_universe``
#: builds every request from ``source_inputs.SUPPORTED_STRATEGIES`` plus the one
#: ``DYN-SV`` meta-request, and also enumerates the disabled CAL-P/CND-P,
#: which the producer refuses before this staging, so they never reach it --
#: exactly the set the candidate policy above (straddle
#: ∩ vs listed put) and ``planned_exit_date``'s exit anchor both know. Sessions:
#: the two ``planned_exit_date`` anchors on. An unsupported nonempty value is
#: malformed input, never a legitimately empty eligible expiry domain.
_SUPPORTED_STRATEGIES = frozenset(SUPPORTED_STRATEGIES) | {"DYN-SV"}
_SUPPORTED_SESSIONS = frozenset({"BMO", "AMC"})
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


def _require_supported_domains(key: BoardRequest) -> None:
    """Strategy/session domain checks, raised before any expiry handling: an
    unsupported nonempty value would otherwise be staged as if it were a real
    request whose listed expiry domain simply came back empty, and land on
    ``NO_RESOLVABLE_EXPIRY`` -- a refusal shape that hides malformed input
    behind a data-shape one. Never echo the submitted value back."""
    if key.strategy not in _SUPPORTED_STRATEGIES:
        raise fail("INVALID_REQUEST", "calendar input key has an unsupported strategy")
    if key.session not in _SUPPORTED_SESSIONS:
        raise fail("INVALID_REQUEST", "calendar input key has an unsupported session")


def _date_string(value: Any) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def _validated_candidate(row: dict) -> tuple[str, str, float]:
    """One projected row's ``(expiry day, normalized right, strike)``, validated
    before any expiry filtering: a null ``expiry``/``strike``/``right``, a right
    outside C/P or a non-finite/unconvertible strike is malformed pinned input
    and refuses ``CONTRACT_MISMATCH`` -- for every row the scan returns,
    including one whose expiry the candidate filter would drop."""
    require_usable_fields(row, check_quotes=False)
    return _date_string(row["expiry"]), str(row["right"]).upper(), float(row["strike"])


def _scan_candidate_rows(repository: Repository, snapshot: SnapshotRef, key: BoardRequest,
                         session_day: str):
    """The per-key bounded scan of one ``(ticker, session)`` slice, projected to
    ``_CANDIDATE_COLUMNS``; yields row dicts."""
    contract = repository.table_contract(snapshot, CHAIN_TABLE)
    version = snapshot.table_versions[CHAIN_TABLE]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(key.ticker,)),
                  KeyPredicate(column="obs_date", operator="eq", values=(session_day,)))
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=CHAIN_TABLE,
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=None)
    max_result_rows = min(_RESULT_CAP, population_bound)
    max_batch_rows = (min(contract.maximum_batch_rows, _BATCH_CAP, max_result_rows)
                      if max_result_rows > 0
                      else min(contract.maximum_batch_rows, _BATCH_CAP))
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=_CANDIDATE_COLUMNS,
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    for batch in repository.scan(query, table_name=CHAIN_TABLE):
        yield from batch.to_pylist()


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
                            decision_session: Any,
                            chain_rows: Sequence[Mapping[str, Any]] | None = None
                            ) -> tuple[str, ...]:
    """One bounded ``option_chains`` scan of the exact ``(ticker, decision_session)``
    slice, projected to raw ``(right, strike, expiry)``: no other session, no
    quote-usability filter, no expiry before the decision session. Every
    projected row's ``expiry``/``strike``/``right`` is validated before any
    expiry filtering -- a malformed one is ``CONTRACT_MISMATCH``. The key's
    strategy and session domains are checked before any scan, so an unsupported
    nonempty value is ``INVALID_REQUEST`` rather than a masked empty expiry
    domain. The eligible domain follows the native strategy policy; the result
    is the sorted distinct ISO tuple, empty only when the domain is empty.
    ``chain_rows`` (that ticker's rows for the session, from one batched build
    read) replaces the scan when given; ``None`` keeps the per-key scan."""
    _require_key(key)
    _require_supported_domains(key)
    session_day = _calendar_day(decision_session, "decision_session")
    rows = (_scan_candidate_rows(repository, snapshot, key, session_day)
            if chain_rows is None
            else require_chain_rows(chain_rows, key.ticker, session_day, check_quotes=False))
    calls: dict[str, set[float]] = {}
    puts: dict[str, set[float]] = {}
    for row in rows:
        expiry_day, right, strike = _validated_candidate(row)
        if expiry_day < session_day:
            continue
        if right == "C":
            calls.setdefault(expiry_day, set()).add(strike)
        else:
            puts.setdefault(expiry_day, set()).add(strike)
    if key.strategy in _STRADDLE_DOMAIN:
        expiries = {day for day in calls if calls[day] & puts.get(day, set())}
    else:
        expiries = set(puts)
    return tuple(sorted(expiries))


def scan_calendar_row_inputs(repository: Repository, snapshot: SnapshotRef, key: BoardRequest, *,
                             decision_session: Any, calendar: CalendarSessions,
                             chain_rows: Sequence[Mapping[str, Any]] | None = None
                             ) -> CalendarRowInputs:
    """One row's pinned spot, resolved strategy expiry and independent planned
    exit staged into ``scan_calendar_row``. A missing or unusable exact-session
    spot raises the per-key ``PRICE_HISTORY_NOT_AVAILABLE`` refusal; a
    repository failure still fails the whole call. The key's strategy and
    session domains are checked first, so an unsupported value is
    ``INVALID_REQUEST`` rather than a masked expiry refusal; an empty eligible
    expiry domain or native no-expiry geometry refusal becomes per-key
    ``NO_RESOLVABLE_EXPIRY``. Missing RUNUP pre-print coverage becomes per-key
    ``RUNUP_HORIZON_NOT_AVAILABLE``; a negative distance becomes per-key
    ``INVALID_RUNUP_HORIZON``. Malformed calendar input and other unhandled
    geometry/input errors propagate."""
    from engine.v2.domain.generation import GeometryRefusal, resolve_expiry

    _require_key(key)
    _require_supported_domains(key)
    _calendar_day(key.event_date, "event_date")
    session_day = _calendar_day(decision_session, "decision_session")
    if chain_rows is not None:
        chain_rows = require_chain_rows(chain_rows, key.ticker, session_day, check_quotes=False)
    series = get_price_series(repository, PriceQuery(
        ticker=key.ticker, session_date=session_day, observation_ceiling=session_day,
        lookback_sessions=0), snapshot)
    spot_row = next((row for row in series if row.date == session_day), None)
    if spot_row is None:
        raise NativeScoreBatchRowRefusal(
            key, "PRICE_HISTORY_NOT_AVAILABLE", PRICE_HISTORY_NOT_AVAILABLE_DETAIL)
    spot = spot_row.close_raw
    if spot is None or not math.isfinite(spot) or spot <= 0:
        raise NativeScoreBatchRowRefusal(
            key, "PRICE_HISTORY_NOT_AVAILABLE", PRICE_HISTORY_NOT_AVAILABLE_DETAIL)
    candidates = scan_candidate_expiries(repository, snapshot, key, decision_session=session_day,
                                         chain_rows=chain_rows)
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
    timing = _runup_timing(key, calendar, session_day)
    exit_date = planned_exit_date(key, calendar)
    row = scan_calendar_row(repository, snapshot, key, entry_date=session_day,
                            exit_date=exit_date, expiry=expiry, spot=spot,
                            calendar_observed_through=calendar.observed_through)
    return replace(row, calendar_row={**row.calendar_row, **timing}) if timing else row


def _runup_timing(key: BoardRequest, calendar: CalendarSessions,
                  entry_date: str) -> dict[str, int]:
    """Stage only RUNUP's calendar-owned horizon, retaining per-key refusals."""
    if key.strategy != "STR-RUNUP":
        return {}
    try:
        days = trading_days_before_print(
            calendar, entry_date=entry_date, event_date=key.event_date, session=key.session)
    except CalendarCoverageError as exc:
        raise NativeScoreBatchRowRefusal(
            key, "RUNUP_HORIZON_NOT_AVAILABLE",
            "the pinned calendar does not cover the RUNUP horizon") from exc
    if days < 0:
        raise NativeScoreBatchRowRefusal(
            key, "INVALID_RUNUP_HORIZON", "RUNUP entry is after the pre-print close")
    return {"days_before_print": days}


def _no_resolvable(key: BoardRequest) -> NativeScoreBatchRowRefusal:
    return NativeScoreBatchRowRefusal(key, _NO_RESOLVABLE, "no strategy-eligible listed expiry")
