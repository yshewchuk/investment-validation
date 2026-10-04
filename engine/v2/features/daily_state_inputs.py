"""Caller-pinned, bounded ``daily_market`` inputs; session-only, no whole-panel anchor."""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from engine.v2.contracts.data import (
    DataQuery,
    DatasetVersionRef,
    KeyPredicate,
    SnapshotRef,
    TableContract,
    TimeInterval,
)
from engine.v2.data import errors, repository
from engine.v2.features import panel_math

__all__ = ["DailyStateInputs", "scan_daily_state_inputs"]
_TABLE = "daily_market"
_BATCH_LIMIT = 1000
_RESULT_LIMIT = 10000


@dataclass(frozen=True, kw_only=True)
class DailyStateInputs:
    """Immutable result: absent values stay absent, eligible session and identity pinned."""

    values: Mapping[str, float]
    source_session: str | None
    snapshot_id: str
    dataset_version_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


def _is_nonblank(value: object) -> bool:
    """True only for a string that is not empty or whitespace alone."""
    return isinstance(value, str) and bool(value.strip())


def _session_day(value: object, field: str) -> pd.Timestamp:
    """One explicit naive midnight calendar day, or a typed refusal."""
    message = f"{field} must be a naive midnight calendar day"
    if isinstance(value, numbers.Number):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field}) from None
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise errors.fail("CONTRACT_MISMATCH", message, details={"field": field})
    return day


def _window(history_start: object, decision_session: object) -> tuple[pd.Timestamp, pd.Timestamp,
                                                                     pd.Timestamp]:
    """``(start, decision, end)``, the half-open day interval this read uses."""
    start = _session_day(history_start, "history_start")
    decision = _session_day(decision_session, "decision_session")
    if start > decision:
        raise errors.fail("CONTRACT_MISMATCH",
                          "history_start must be on or before decision_session")
    try:
        end = decision + pd.Timedelta(days=1)
    except (OverflowError, ValueError):
        raise errors.fail("CONTRACT_MISMATCH",
                          "decision_session is too far in the future to bound") from None
    return start, decision, end


def _resolve_snapshot(data_repository: repository.Repository, snapshot: SnapshotRef) -> None:
    """Refuse before any read unless the supplied ref is the resolved one."""
    if not isinstance(snapshot, SnapshotRef) or not _is_nonblank(snapshot.snapshot_id):
        raise errors.fail("CONTRACT_MISMATCH", "snapshot must carry a nonblank id")
    resolved = data_repository.resolve(snapshot.snapshot_id)
    if resolved != snapshot:
        raise errors.fail("CONTRACT_MISMATCH",
                          "supplied snapshot does not match the resolved snapshot",
                          details={"snapshot_id": snapshot.snapshot_id})


def _daily_version(snapshot: SnapshotRef) -> DatasetVersionRef:
    """The pinned ``daily_market`` version handle, present and nonblank."""
    version = snapshot.table_versions.get(_TABLE)
    if version is None or not _is_nonblank(version.dataset_version_id):
        raise errors.fail("CONTRACT_MISMATCH",
                          "daily_market dataset version is missing or blank",
                          details={"table_name": _TABLE})
    return version


def _build_query(data_repository: repository.Repository, snapshot: SnapshotRef,
                 version: DatasetVersionRef, contract: TableContract,
                 ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> DataQuery:
    """The one bounded single-ticker read this module ever issues."""
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    time_interval = TimeInterval(
        column="date",
        start_inclusive=start.date().isoformat(),
        end_exclusive=end.date().isoformat(),
    )
    max_batch_rows = min(contract.maximum_batch_rows, _BATCH_LIMIT)
    max_result_rows = min(contract.maximum_result_rows, _RESULT_LIMIT)
    population_bound = data_repository.scan_population_bound(
        snapshot.snapshot_id, table_name=_TABLE,
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=time_interval)
    if population_bound > 0:
        # Only a positive selected bound lowers the active result limit; a
        # zero bound keeps it positive (zero-result queries arrive with slice E).
        max_result_rows = min(max_result_rows, population_bound)
    if max_result_rows > 0:
        # A lowered positive result limit may not leave the batch limit above it
        # (BATCH_EXCEEDS_RESULT is refused at decode): lower only as needed.
        max_batch_rows = min(max_batch_rows, max_result_rows)
    return DataQuery(
        snapshot_id=snapshot.snapshot_id,
        table_contract_ref=version.table_contract_ref,
        columns=("ticker", "date", "src_iv", *panel_math.DAILY_STATE_FIELDS),
        key_filter=key_filter,
        time_interval=time_interval,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows,
    )


def _validated_row(row: object, ticker: str, start: pd.Timestamp,
                   end: pd.Timestamp) -> dict[str, object]:
    """One copied, normalized row, or a typed refusal naming only the table."""
    if not isinstance(row, Mapping):
        raise errors.fail("CONTRACT_MISMATCH", "daily_market row is not a mapping",
                          details={"table_name": _TABLE})
    if row.get("ticker") != ticker:
        raise errors.fail("CONTRACT_MISMATCH",
                          "daily_market row ticker does not match the requested ticker",
                          details={"table_name": _TABLE})
    day = _session_day(row.get("date"), "daily_market.date")
    if day < start or day >= end:
        raise errors.fail("CONTRACT_MISMATCH",
                          "daily_market row date is outside the requested window",
                          details={"table_name": _TABLE})
    validated = dict(row)
    validated["date"] = day
    return validated


def _scan_rows(data_repository: repository.Repository, query: DataQuery, ticker: str,
               start: pd.Timestamp, end: pd.Timestamp) -> list[Mapping[str, object]]:
    """Consume the whole bounded scan before returning any row."""
    rows: list[Mapping[str, object]] = []
    seen: set[pd.Timestamp] = set()
    for batch in data_repository.scan(query, table_name=_TABLE):
        for row in batch.to_pylist():
            validated = _validated_row(row, ticker, start, end)
            if len(rows) >= query.max_result_rows:
                raise errors.fail("RESULT_LIMIT_EXCEEDED",
                                  "daily_market scan exceeded the bounded result cap",
                                  details={"table_name": _TABLE})
            if validated["date"] in seen:
                raise errors.fail("IDENTITY_CONFLICT",
                                  "duplicate daily_market date for the requested ticker",
                                  details={"table_name": _TABLE})
            seen.add(validated["date"])
            rows.append(validated)
    return rows


def _inputs_from_rows(rows: list[Mapping[str, object]],
                      decision: pd.Timestamp) -> tuple[dict[str, float], str | None]:
    """The lookup mapping and the selected eligible source session."""
    try:
        values = panel_math.daily_state_lookup(rows, decision)
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "daily_market value conversion failed",
                          details={"table_name": _TABLE}) from None
    eligible = [row for row in rows if panel_math._is_present(row.get("src_iv"))]
    if not eligible:
        return values, None
    session = max(eligible, key=lambda row: row["date"])["date"]
    return values, session.date().isoformat()


def scan_daily_state_inputs(
    repository: repository.Repository,
    snapshot: SnapshotRef,
    *,
    ticker: str,
    history_start: object,
    decision_session: object,
) -> DailyStateInputs:
    """One ticker's caller-pinned ``daily_market`` state; no intraday or panel anchor."""
    if not _is_nonblank(ticker):
        raise errors.fail("CONTRACT_MISMATCH", "ticker must be a nonblank string")
    start, decision, end = _window(history_start, decision_session)
    _resolve_snapshot(repository, snapshot)
    contract = repository.table_contract(snapshot, _TABLE)
    version = _daily_version(snapshot)
    query = _build_query(repository, snapshot, version, contract, ticker, start, end)
    rows = _scan_rows(repository, query, ticker, start, end)
    values, source_session = _inputs_from_rows(rows, decision)
    return DailyStateInputs(
        values=values,
        source_session=source_session,
        snapshot_id=snapshot.snapshot_id,
        dataset_version_id=version.dataset_version_id,
    )
