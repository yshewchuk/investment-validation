"""Pinned-snapshot reads shared by the ``engine/v2/research`` tools.

One ``DataQuery`` per manifest partition, bounded by that partition's own
fragment ``time_min``/``time_max`` records — the same manifest-derived rule
``engine.v2.data.legacy_materialization`` applies to a whole-table read, never
a sentinel bound invented out of thin air. Every read therefore stays inside
``Repository.scan``'s bounded-scan contract (§8.2): no ``read_table()``
convenience, no implicit "latest".

The table contract's ``maximum_result_rows`` caps ONE partition's scan. A
partition whose manifest row count exceeds its own table's cap refuses loudly
with ``RESULT_LIMIT_EXCEEDED`` rather than silently truncating; partition the
table more finely in that case. The shipped Tier-2 tables are partitioned by
year and this suffices for them.

Internal to the package: nothing here is part of ``engine.v2.research``'s
public interface.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

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
               *, partition_keys=None) -> pd.DataFrame:
    """Every row of ``table_name`` in ``snapshot_ref``, projected to ``columns``.

    ``partition_keys`` restricts the read to those manifest partitions (the
    Tier-2 tables are partitioned by year). A table absent from the snapshot,
    or one whose fragments carry no recorded time bounds, refuses with
    ``CONTRACT_MISMATCH`` rather than scanning a guess.
    """
    if table_name not in snapshot_ref.table_versions:
        raise errors.fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                          details={"table_name": table_name})
    contract = repository.table_contract(snapshot_ref, table_name)
    records = list(repository.fragment_records(snapshot_ref, table_name))
    if partition_keys is not None:
        wanted = {str(key) for key in partition_keys}
        records = [record for record in records if record.partition_key in wanted]
    rows: list[dict] = []
    for partition in dict.fromkeys(record.partition_key for record in records):
        group = [record for record in records if record.partition_key == partition]
        rows.extend(_scan_partition(repository, snapshot_ref, table_name, contract, columns, group))
    return pd.DataFrame(rows, columns=list(columns))


def _scan_partition(repository, snapshot_ref: SnapshotRef, table_name: str, contract,
                    columns, records: list) -> list[dict]:
    interval = _partition_interval(contract, table_name, records)
    row_bound = max(1, sum(record.row_count for record in records))
    result_cap = min(row_bound, contract.maximum_result_rows)
    batch_cap = min(result_cap, contract.maximum_batch_rows)
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id,
        table_contract_ref=snapshot_ref.table_versions[table_name].table_contract_ref,
        columns=tuple(columns), key_filter=(), time_interval=interval,
        order_by=contract.primary_key, max_batch_rows=batch_cap, max_result_rows=result_cap)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=table_name):
        rows.extend(batch.to_pylist())
    return rows


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
