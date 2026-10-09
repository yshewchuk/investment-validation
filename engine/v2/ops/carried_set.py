"""Carried-set resolution over one pinned snapshot.

A ticker is carried only when the snapshot holds at least one ``daily_market.date``
row AND at least one ``option_chains.obs_date`` row, each inside the inclusive
window from January 1 of ``as_of.year - 1`` through ``as_of``. Both reads stay
pinned to the passed ``SnapshotRef`` — never a mutable head — and every reader
exception propagates unchanged: an unreadable table is never an empty ticker set.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from engine.v2.contracts import DataQuery, SnapshotRef, TimeInterval
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail

__all__ = ["CarriedTickerExclusion", "CarriedSetResolution",
           "resolve_carried_set", "build_uncarried_exclusions"]

_DAILY_TABLE = "daily_market"
_CHAINS_TABLE = "option_chains"

#: The exact row fields this read takes from each source table.
_TABLE_FIELDS = {
    _DAILY_TABLE: ("ticker", "date"),
    _CHAINS_TABLE: ("ticker", "obs_date"),
}


@dataclass(frozen=True, slots=True)
class CarriedTickerExclusion:
    ticker: str
    reason_code: Literal["UNCARRIED_TICKER"]
    missing_tables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CarriedSetResolution:
    daily_market_tickers: tuple[str, ...]
    option_chain_tickers: tuple[str, ...]

    @property
    def carried_tickers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.daily_market_tickers) & set(self.option_chain_tickers)))


def _parse_as_of(as_of: str) -> date:
    try:
        parsed = date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST", "as_of must be a canonical ISO date") from None
    if parsed.isoformat() != as_of:
        raise fail("INVALID_REQUEST", "as_of must be a canonical ISO date")
    return parsed


def _day_string(value: object) -> str:
    """Normalize an Arrow-delivered datetime/date value or ISO string to a day string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _scan_tickers(repository: Repository, snapshot: SnapshotRef, table_name: str,
                  interval: TimeInterval, start_day: str, as_of_day: str) -> tuple[str, ...]:
    """One bounded, pinned, whole-window scan of a table's (ticker, date) rows."""
    ticker_column, date_column = _TABLE_FIELDS[table_name]
    contract = repository.table_contract(snapshot, table_name)
    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name,
        table_contract_ref=contract_ref, time_interval=interval)
    batch_rows = min(contract.maximum_batch_rows, bound) if bound > 0 else contract.maximum_batch_rows
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=(ticker_column, date_column), key_filter=(),
        time_interval=interval, order_by=tuple(contract.primary_key),
        max_batch_rows=batch_rows, max_result_rows=bound)
    tickers: set[str] = set()
    for batch in repository.scan(query, table_name=table_name):
        for row in batch.to_pylist():
            if start_day <= _day_string(row[date_column]) <= as_of_day:
                tickers.add(str(row[ticker_column]))
    return tuple(sorted(tickers))


def resolve_carried_set(repository: Repository, snapshot: SnapshotRef, *,
                        as_of: str) -> CarriedSetResolution:
    """The per-table ticker sets the pinned snapshot carries as of ``as_of``."""
    session = _parse_as_of(as_of)
    start_day = f"{session.year - 1}-01-01"
    as_of_day = session.isoformat()
    sets: dict[str, tuple[str, ...]] = {}
    for table_name, (_ticker_column, date_column) in _TABLE_FIELDS.items():
        interval = TimeInterval(column=date_column, start_inclusive=start_day,
                                end_exclusive=(session + timedelta(days=1)).isoformat())
        sets[table_name] = _scan_tickers(repository, snapshot, table_name, interval,
                                         start_day, as_of_day)
    return CarriedSetResolution(daily_market_tickers=sets[_DAILY_TABLE],
                                option_chain_tickers=sets[_CHAINS_TABLE])


def build_uncarried_exclusions(candidate_tickers: Iterable[str],
                               carried: CarriedSetResolution) -> tuple[CarriedTickerExclusion, ...]:
    """One sorted exclusion per deduplicated candidate that is not carried."""
    present = {_DAILY_TABLE: set(carried.daily_market_tickers),
               _CHAINS_TABLE: set(carried.option_chain_tickers)}
    exclusions = []
    for ticker in sorted(set(candidate_tickers)):
        missing = tuple(sorted(name for name, tickers in present.items() if ticker not in tickers))
        if missing:
            exclusions.append(CarriedTickerExclusion(ticker=ticker,
                                                     reason_code="UNCARRIED_TICKER",
                                                     missing_tables=missing))
    return tuple(exclusions)
