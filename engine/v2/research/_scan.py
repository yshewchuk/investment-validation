"""Pinned-snapshot reads shared by the ``engine/v2/research`` tools.

One ``DataQuery`` per manifest partition, scoped to its partition key and the
caller's key predicates, without an observation-time interval. The selected
fragment membership bound lets each read return the complete partition,
including null and non-midnight observation times. Every read stays inside
``Repository.scan``'s bounded-scan contract (§8.2), never an implicit latest.

The manifest population bound is an upper bound on candidate rows, not process
memory. ``batch_filter`` runs on each pandas batch before accumulation and may
only narrow retained rows. A typed scan failure returns no partial frame.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from engine.v2.contracts.data import DataQuery, KeyPredicate, SnapshotRef, TimeInterval
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
               *, partition_keys=None, key_filter=(), batch_filter=None) -> pd.DataFrame:
    """Every row of ``table_name`` in ``snapshot_ref``, projected to ``columns``.

    ``partition_keys`` restricts the read to those manifest partitions (the
    Tier-2 tables are partitioned by year). ``key_filter`` is a tuple of
    ``KeyPredicate`` values carried into every scan, so a caller that needs
    only specific keys reads only rows that can match them. ``batch_filter``,
    when given, is called on each batch frame right after ``to_pandas()`` and
    before it accumulates, and must return the frame narrowed to the rows the
    caller wants (``None`` may be returned for "keep nothing"); a ``None`` or
    omitted ``batch_filter`` changes nothing. Every scan is bounded by its own
    pinned membership bound, so a complete partition is read fully. A table
    absent from the snapshot refuses with ``CONTRACT_MISMATCH``; selected
    partitions are scanned in full without an observation-time interval.
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
                                      group, key_filter=tuple(key_filter),
                                      batch_filter=batch_filter))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _scan_partition(repository, snapshot_ref: SnapshotRef, table_name: str, contract,
                    columns, records: list, *, key_filter,
                    batch_filter=None) -> list[pd.DataFrame]:
    """Read one manifest partition completely in a single bounded scan."""
    partition_filter = _partition_filter(contract, table_name, records)
    predicates = tuple(key_filter)
    if partition_filter is not None:
        predicates = (*predicates, partition_filter)
    return _scan_interval(repository, snapshot_ref, table_name, contract, columns,
                          predicates, None, batch_filter=batch_filter)


def _partition_filter(contract, table_name: str, records: list) -> KeyPredicate | None:
    """An equality ``KeyPredicate`` scoping a scan to ``records``' own partition.

    Every current caller's partition column is the single integer ``"year"``
    (``records[0].partition_key`` is its decimal text), so one ``eq``
    predicate on that converted value is enough. A contract with no declared
    partition column — never a real research read (``_snapshot.read_table``
    refuses those) — has nothing to scope by and returns ``None``; more than
    one partition column, or a key this rule cannot cleanly convert, would
    silently scope by the wrong thing, so both refuse instead.
    """
    partition_columns = getattr(contract, "partition_columns", ())
    if not partition_columns:
        return None
    if len(partition_columns) > 1:
        raise errors.fail("CONTRACT_MISMATCH",
                          "a scoped research read supports one partition column, not "
                          f"{len(partition_columns)}",
                          details={"table_name": table_name})
    column = partition_columns[0]
    try:
        value = int(records[0].partition_key)
    except (TypeError, ValueError):
        raise errors.fail("CONTRACT_MISMATCH",
                          f"partition column {column!r} is not an integer partition key",
                          details={"table_name": table_name,
                                   "partition_key": records[0].partition_key}) from None
    return KeyPredicate(column=column, operator="eq", values=(value,))


def _scan_interval(repository, snapshot_ref: SnapshotRef, table_name: str, contract,
                   columns, key_filter, interval: TimeInterval | None,
                   *, batch_filter=None) -> list[pd.DataFrame]:
    """One bounded scan over ``interval`` (or the whole partition when ``None``).

    The query's result bound is exactly ``Repository.scan_population_bound``
    for this snapshot, table, pinned contract ref, predicate set and interval —
    the recorded row counts of the fragments that selection survives, zero for
    an empty membership — and the batch limit follows it: at most the
    contract's ``maximum_batch_rows``, lowered by a positive result bound and
    kept positive when the bound is zero.

    Each Arrow batch becomes pandas, passes through ``batch_filter`` when one
    is given, and only the surviving (non-empty) frames are returned — so rows
    the filter rejects never outlive their batch, and batches that match
    nothing contribute nothing. Frames stay local to this call: a typed failure
    after batches were yielded discards them, and the failure propagates without
    retry. The scan's own error contracts are unchanged, and bound planning adds
    no retry: a planning ``DataError`` propagates exactly as raised.
    """
    predicates = tuple(key_filter)
    contract_ref = snapshot_ref.table_versions[table_name].table_contract_ref
    max_result_rows = repository.scan_population_bound(
        snapshot_ref.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=predicates, time_interval=interval)
    max_batch_rows = (min(contract.maximum_batch_rows, max_result_rows)
                      if max_result_rows > 0 else contract.maximum_batch_rows)
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id,
        table_contract_ref=contract_ref,
        columns=tuple(columns), key_filter=predicates, time_interval=interval,
        order_by=contract.primary_key, max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    frames: list[pd.DataFrame] = []
    for batch in repository.scan(query, table_name=table_name):
        frame = batch.to_pandas()
        if batch_filter is not None:
            frame = batch_filter(frame)
        if frame is not None and not frame.empty:
            frames.append(frame)
    return frames
