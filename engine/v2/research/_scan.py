"""Pinned-snapshot reads shared by the ``engine/v2/research`` tools.

One ``DataQuery`` per manifest partition, bounded by that partition's own
fragment ``time_min``/``time_max`` records — the same manifest-derived rule
``engine.v2.data.legacy_materialization`` applies to a whole-table read, never
a sentinel bound invented out of thin air. Every read therefore stays inside
``Repository.scan``'s bounded-scan contract (§8.2): no ``read_table()``
convenience, no implicit "latest".

The table contract's ``maximum_result_rows`` caps ONE scan, and a real year
partition can exceed it (``option_chains`` carries 2.1M-4.3M rows in every
year 2018-2026 against a 2,000,000 cap). Each partition is therefore scanned
as its calendar months, and a month that still exceeds the cap is scanned as
its days; only a single day that still exceeds the cap refuses loudly with
``RESULT_LIMIT_EXCEEDED`` rather than silently truncating. Every split keeps
the one ``snapshot_id`` the caller resolved.

A caller's ``key_filter`` is threaded into every one of those scans, so a
reader that needs only specific keys (``fill_quality`` needs only the traded
contracts' chain rows) never reads a whole table to discard most of it.

Frames are assembled with ``batch.to_pandas()`` per Arrow batch and one
``pd.concat``; converting each batch to a list of Python dicts first would
cost multiples of the frame it produces, so this module does not.

Internal to the package: nothing here is part of ``engine.v2.research``'s
public interface.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from engine.v2.contracts.data import DataQuery, SnapshotRef, TimeInterval
from engine.v2.data import errors, time_formats

__all__ = ["DEFAULT_SCOPE", "next_representable", "read_table", "resolve_snapshot"]

#: The effect scope a full nightly run writes (``engine.v2.ops.nightly``).
DEFAULT_SCOPE = "shadow"


def resolve_snapshot(repository, *, scope: str = DEFAULT_SCOPE,
                     snapshot_id: str | None = None) -> SnapshotRef:
    """The one snapshot a run reads: an explicit ``snapshot_id``, or ``scope``'s
    pinned head. Exactly one call, made once per run by each tool's ``run``."""
    if snapshot_id is not None:
        return repository.resolve(snapshot_id)
    return repository.resolve_pinned(scope)


def next_representable(value: str) -> str:
    """The smallest value strictly after ``value``, in ``value``'s own
    encoding: a naive timestamp advances one microsecond, a bare date one day."""
    if time_formats.is_naive_timestamp(value):
        parsed = datetime.strptime(value, time_formats.NAIVE_TIMESTAMP_FORMAT)
        return time_formats.format_naive_timestamp(parsed + timedelta(microseconds=1))
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def read_table(repository, snapshot_ref: SnapshotRef, table_name: str, columns,
               *, partition_keys=None, key_filter=()) -> pd.DataFrame:
    """Every row of ``table_name`` in ``snapshot_ref``, projected to ``columns``.

    ``partition_keys`` restricts the read to those manifest partitions (the
    Tier-2 tables are partitioned by year). ``key_filter`` is a tuple of
    ``KeyPredicate`` values carried into every scan, so a caller that needs
    only specific keys reads only rows that can match them. A table absent
    from the snapshot, or one whose fragments carry no recorded time bounds,
    refuses with ``CONTRACT_MISMATCH`` rather than scanning a guess.
    """
    if table_name not in snapshot_ref.table_versions:
        raise errors.fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                          details={"table_name": table_name})
    contract = repository.table_contract(snapshot_ref, table_name)
    records = list(repository.fragment_records(snapshot_ref, table_name))
    if partition_keys is not None:
        wanted = {str(key) for key in partition_keys}
        records = [record for record in records if record.partition_key in wanted]
    frames: list[pd.DataFrame] = []
    for partition in dict.fromkeys(record.partition_key for record in records):
        group = [record for record in records if record.partition_key == partition]
        frames.extend(_scan_partition(repository, snapshot_ref, table_name, contract, columns,
                                      group, key_filter=tuple(key_filter)))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _scan_partition(repository, snapshot_ref: SnapshotRef, table_name: str, contract,
                    columns, records: list, *, key_filter) -> list[pd.DataFrame]:
    """One partition's rows: a scan per calendar month, its days on overflow.

    A month whose row count exceeds the contract's ``maximum_result_rows``
    raises ``RESULT_LIMIT_EXCEEDED`` from the repository; that month is then
    rescanned one day at a time. A day that still exceeds the cap propagates
    the error — the table needs finer partitions than this rule can supply.
    """
    interval = _partition_interval(contract, table_name, records)
    frames: list[pd.DataFrame] = []
    for month in _calendar_intervals(interval, "month"):
        try:
            frames.append(_scan_interval(repository, snapshot_ref, table_name, contract,
                                         columns, key_filter, month))
        except errors.DataError as exc:
            if exc.code != "RESULT_LIMIT_EXCEEDED":
                raise
            days = _calendar_intervals(month, "day")
            if not days:
                raise
            for day in days:
                frames.append(_scan_interval(repository, snapshot_ref, table_name, contract,
                                             columns, key_filter, day))
    return frames


def _scan_interval(repository, snapshot_ref: SnapshotRef, table_name: str, contract,
                   columns, key_filter, interval: TimeInterval) -> pd.DataFrame:
    """One bounded scan over ``interval``, as a frame of ``columns``."""
    result_cap = contract.maximum_result_rows
    batch_cap = min(contract.maximum_batch_rows, result_cap)
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id,
        table_contract_ref=snapshot_ref.table_versions[table_name].table_contract_ref,
        columns=tuple(columns), key_filter=tuple(key_filter), time_interval=interval,
        order_by=contract.primary_key, max_batch_rows=batch_cap, max_result_rows=result_cap)
    batches = [batch.to_pandas() for batch in repository.scan(query, table_name=table_name)]
    if not batches:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(batches, ignore_index=True)


def _calendar_intervals(interval: TimeInterval, step: str) -> list[TimeInterval]:
    """``interval`` cut at calendar boundaries (``"month"`` or ``"day"``),
    keeping the half-open shape and the source bound's own encoding."""
    start = _parse_bound(interval.start_inclusive)
    end = _parse_bound(interval.end_exclusive)
    if start is None or end is None or start >= end:
        return []
    wire = (time_formats.is_naive_timestamp(interval.start_inclusive)
            or time_formats.is_naive_timestamp(interval.end_exclusive))
    intervals: list[TimeInterval] = []
    cursor = start
    while cursor < end:
        stop = min(_next_boundary(cursor, step), end)
        intervals.append(TimeInterval(
            column=interval.column,
            start_inclusive=_format_bound(cursor, wire),
            end_exclusive=_format_bound(stop, wire)))
        cursor = stop
    return intervals


def _next_boundary(value: datetime, step: str) -> datetime:
    """The next calendar boundary strictly after ``value`` for ``step``."""
    if step == "day":
        return value + timedelta(days=1)
    return (value.replace(day=1) + timedelta(days=32)).replace(day=1)


def _parse_bound(value: str | None) -> datetime | None:
    """A recorded time bound as a naive UTC datetime (bare dates at midnight)."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _format_bound(value: datetime, wire: bool) -> str:
    """A boundary back in the interval's own encoding: naive-timestamp wire
    form when the partition bounds use it, a bare date otherwise."""
    if wire:
        return time_formats.format_naive_timestamp(value)
    return value.date().isoformat()


def _partition_interval(contract, table_name: str, records: list) -> TimeInterval:
    column = contract.observation_time_column
    if not column:
        raise errors.fail("CONTRACT_MISMATCH",
                          "a research read requires an observation_time_column",
                          details={"table_name": table_name})
    minima = [record.time_min for record in records if record.time_min is not None]
    maxima = [record.time_max for record in records if record.time_max is not None]
    if not minima or not maxima:
        raise errors.fail("CONTRACT_MISMATCH",
                          "a research read needs recorded fragment time bounds",
                          details={"table_name": table_name})
    return TimeInterval(column=column, start_inclusive=min(minima),
                        end_exclusive=next_representable(max(maxima)))
