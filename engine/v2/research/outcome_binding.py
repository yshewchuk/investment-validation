"""Native source-bound identity and authentication for isolated outcomes:
anchors, computed-move fragment index and exact singleton binding."""
from datetime import date

from engine.v2.contracts.data import DataQuery, KeyPredicate
from engine.v2.data.computed_move_bounds import CONTRACT as BOUNDS_CONTRACT
from engine.v2.data.computed_move_bounds import TABLE as BOUNDS_TABLE
from engine.v2.data.errors import fail


def anchors(repository, snapshot, limits, required_tickers=()):
    records = repository.fragment_records(snapshot, BOUNDS_TABLE)
    if set(required_tickers) - {r.partition_key for r in records}:
        raise fail("CONTRACT_MISMATCH", "required ticker is missing native source bounds")
    count = sum(r.row_count for r in records)
    if not count:
        return {}
    request = DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions[BOUNDS_TABLE].table_contract_ref,
        columns=tuple(c.name for c in BOUNDS_CONTRACT.columns), order_by=BOUNDS_CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="ticker", operator="in",
            values=tuple(sorted({r.partition_key for r in records}))),),
        max_batch_rows=min(count, BOUNDS_CONTRACT.maximum_batch_rows), max_result_rows=count)
    result = {}
    for batch in repository.scan(request, table_name=BOUNDS_TABLE, max_object_bytes=limits["max_object_bytes"]):
        for row in batch.to_pylist():
            key = row["ticker"], row["event_date"]
            if key in result:
                raise fail("CONTRACT_MISMATCH", "duplicate source bounds")
            result[key] = row
    if len(result) != count:
        raise fail("CONTRACT_MISMATCH", "incomplete source bounds")
    return result


def bound(event, anchors, records):
    """Authenticate metadata to exact singleton object without opening that object."""
    key = event.ticker, event.event_date.date().isoformat()
    row, record = anchors.get(key), records.get(key)
    if row is None:
        raise fail("CONTRACT_MISMATCH", "required event is missing native source bounds")
    if (record is None or record.row_count != 1
            or record.primary_key_min != key or record.primary_key_max != key
            or record.partition_key != key[0] or record.time_min != key[1] or record.time_max != key[1]
            or row["outcome_object_id"] != record.object_ref.object_id
            or row["outcome_content_hash"] != record.object_ref.content_hash):
        raise fail("HOLDOUT_ACCESS_DENIED", "source bounds do not bind the exact isolated outcome")
    return _bound_dates(row, event, key)


def _bound_dates(row, event, key):
    """Null means unavailable; non-null dates must be the valid source bracket."""
    lo, hi = row["pre_close_date"], row["post_close_date"]
    if lo is None and hi is None:
        return None
    try:
        left, right = date.fromisoformat(lo), date.fromisoformat(hi)
        if (left.isoformat() != lo or right.isoformat() != hi or not lo <= key[1] <= hi
                or not 0 < (right - left).days <= 5 or row["session"] != event.session):
            raise ValueError
    except (TypeError, ValueError):
        raise fail("CONTRACT_MISMATCH", "source label bounds are invalid") from None
    return row


def fragment_index(repository, snapshot):
    records = {}
    for record in repository.fragment_records(snapshot, "computed_moves"):
        key = record.primary_key_min
        if key in records:
            raise fail("HOLDOUT_ACCESS_DENIED", "ambiguous outcome object identity")
        records[key] = record
    return records
