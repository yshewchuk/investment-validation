"""Pinned board enumeration and calendar-row staging for cutover PR-6.

Calendar identity comes from the snapshot; market context is explicitly
staged by the caller. Panel/Tier-4/quote sourcing and production wiring
remain separate slices.

The scan deliberately applies no ``src_orats`` filter (unlike
``computed_moves_store._scan_once``, whose backward-looking selection has its
own reason to): ``engine/data/schemas.py``'s ``EARNINGS_EVENTS`` docstring
names ORATS/oquants history-only sources, while Nasdaq/yfinance carry the
forward dates the monitoring board scores.

Like ``native_board_universe``, this module never imports ``engine.score`` /
``engine.structures`` / ``engine.replay`` / ``engine.fills``.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail
from engine.v2.ops.native_board_universe import (
    BoardRequest,
    _validated_as_of,
    _validated_horizon_days,
    board_requests,
)
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal, validated_as_of

__all__ = ["CalendarRowInputs", "scan_calendar_row", "scan_forward_board_requests"]

_EVENTS_TABLE = "earnings_events"
_EVENTS_COLUMNS = ("ticker", "event_date", "session")


def scan_forward_board_requests(
    repository: Repository,
    snapshot: SnapshotRef,
    *,
    as_of,
    horizon_days: int,
    tickers=None,
) -> tuple[BoardRequest, ...]:
    """Scan ``earnings_events`` off ``snapshot`` through ``repository`` -- the
    same read ``computed_moves_store._scan_once`` uses for the same table --
    and return ``native_board_universe.board_requests``'s ordered tuple for
    the forward window. As this module's own docstring says, the scan adds no
    ``src_orats`` filter of its own. It restricts the scanned partitions to
    the years the forward window can touch, re-validating ``as_of``/
    ``horizon_days`` the same way ``board_requests`` does for that reason.

    Outcomes: ``earnings_events`` missing from the pinned snapshot raises
    ``CONTRACT_MISMATCH`` (``Repository.table_contract``'s own typed refusal);
    a forward window with no matching partition year returns an empty tuple,
    not an exception; a malformed ``as_of``/``horizon_days`` raises
    ``INVALID_REQUEST`` (``native_board_universe``'s own validators, run
    before the scan).
    """
    as_of_ts = _validated_as_of(as_of).normalize()
    horizon_days = _validated_horizon_days(horizon_days)
    window_years = set(range(as_of_ts.year, (as_of_ts + pd.Timedelta(days=horizon_days)).year + 1))
    contract = repository.table_contract(snapshot, _EVENTS_TABLE)
    contract_ref = snapshot.table_versions[_EVENTS_TABLE].table_contract_ref
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, _EVENTS_TABLE)}
                         & window_years))
    if not years:
        return board_requests(as_of, horizon_days, tickers,
                              pd.DataFrame(columns=_EVENTS_COLUMNS))
    key_filter = (KeyPredicate(column="year", operator="in", values=years),)
    max_result_rows = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=_EVENTS_TABLE, table_contract_ref=contract_ref,
        key_filter=key_filter)
    max_batch_rows = (min(contract.maximum_batch_rows, max_result_rows)
                      if max_result_rows > 0 else contract.maximum_batch_rows)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=_EVENTS_COLUMNS,
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=_EVENTS_TABLE):
        rows.extend(batch.to_pylist())
    events_table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EVENTS_COLUMNS)
    return board_requests(as_of, horizon_days, tickers, events_table)


@dataclass(frozen=True, slots=True)
class CalendarRowInputs:
    """A staged row and the earnings dataset revision used by EventRef."""

    calendar_revision: str
    calendar_row: Mapping[str, Any]


def _calendar_day(value: Any) -> str:
    """Reject non-day values without exposing submitted text in refusals."""
    try:
        day = validated_as_of(value)
    except NightlySourceBundleRefusal:
        raise fail("INVALID_REQUEST", "calendar context requires valid naive dates") from None
    if day != day.normalize():
        raise fail("INVALID_REQUEST", "calendar context requires midnight dates")
    return day.date().isoformat()


def _calendar_spot(value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, numbers.Number)
            or isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real)):
        raise fail("INVALID_REQUEST", "calendar spot requires a positive finite number")
    try:
        spot = float(value)
    except (TypeError, ValueError, OverflowError):
        raise fail("INVALID_REQUEST", "calendar spot requires a positive finite number") from None
    if not math.isfinite(spot) or spot <= 0:
        raise fail("INVALID_REQUEST", "calendar spot requires a positive finite number")
    return spot


def _pinned_calendar_event(repository: Repository, snapshot: SnapshotRef,
                           key: BoardRequest, event_date: str) -> tuple[str, dict]:
    contract = repository.table_contract(snapshot, _EVENTS_TABLE)
    version = snapshot.table_versions[_EVENTS_TABLE]
    revision = version.dataset_version_id
    if not isinstance(revision, str) or not revision.strip():
        raise fail("INVALID_REQUEST", "earnings dataset revision is missing")
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(key.ticker,)),
                  KeyPredicate(column="year", operator="eq", values=(int(event_date[:4]),)))
    max_result_rows = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=_EVENTS_TABLE,
        table_contract_ref=version.table_contract_ref, key_filter=key_filter)
    max_batch_rows = (min(contract.maximum_batch_rows, max_result_rows)
                      if max_result_rows > 0 else contract.maximum_batch_rows)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=("event_id", *_EVENTS_COLUMNS),
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    matches = []
    target_date = pd.Timestamp(event_date)
    for batch in repository.scan(query, table_name=_EVENTS_TABLE):
        for row in batch.to_pylist():
            if (row["ticker"] == key.ticker and row["event_date"] == target_date
                    and row["session"] == key.session):
                matches.append(row)
    if not matches:
        raise data_fail("EVENT_NOT_FOUND", "no exact calendar event in the pinned snapshot")
    if len(matches) != 1:
        raise data_fail("IDENTITY_CONFLICT", "multiple exact calendar events in the pinned snapshot")
    row = matches[0]
    if not isinstance(row["event_id"], str) or not row["event_id"].strip():
        raise fail("INVALID_REQUEST", "persisted calendar event identity is missing")
    return revision, row


def scan_calendar_row(
    repository: Repository,
    snapshot: SnapshotRef,
    key: BoardRequest,
    *,
    entry_date: Any,
    exit_date: Any,
    expiry: Any,
    spot: Any,
    calendar_observed_through: Any,
) -> CalendarRowInputs:
    """Build one calendar row from a pinned event and staged market context.

    Dates must be naive calendar days and spot a finite positive number.
    No strategy window, expiry or market-price policy is chosen here. The
    downstream SourceBundle assembler checks observation dates against as_of.
    Missing/ambiguous events raise EVENT_NOT_FOUND/IDENTITY_CONFLICT;
    malformed inputs raise INVALID_REQUEST; repository failures propagate.
    """
    if not isinstance(key, BoardRequest) or any(
        not isinstance(value, str) or not value.strip()
        for value in (key.ticker, key.strategy, key.session)
    ):
        raise fail("INVALID_REQUEST", "calendar key requires non-empty identity fields")
    event_date = _calendar_day(key.event_date)
    context = {
        "entry_date": _calendar_day(entry_date), "exit_date": _calendar_day(exit_date),
        "expiry": _calendar_day(expiry), "spot": _calendar_spot(spot),
        "calendar_observed_through": _calendar_day(calendar_observed_through),
    }
    revision, row = _pinned_calendar_event(repository, snapshot, key, event_date)
    return CalendarRowInputs(calendar_revision=revision, calendar_row={
        "event_id": row["event_id"], "ticker": row["ticker"],
        "event_date": event_date, "session": row["session"], **context,
    })
