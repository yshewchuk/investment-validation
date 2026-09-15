"""Pure ``DataQuery`` validation, fragment pruning, and row predicate matching
— phase-2 guide §5.3, §8.2 steps 3 and 5. No I/O: this module never opens a
file or a database connection. ``repository.py`` is the orchestrator that
feeds this module's functions real ``FragmentRecord``/``TableContract``
values and real Arrow-decoded rows; every function here is a pure value
transform, independently testable without a catalog or a store.

Judgement calls (this package's own, task brief P2-4):

* :func:`validate_query` checks ``order_by == contract.primary_key`` exactly
  (guide §5.3 "In v1, order_by must equal the full primary key"), never
  ``contract.orderable_columns`` — ``earnings_events`` deliberately declares
  an ``orderable_columns`` that differs from its primary key, and physical
  fragment rows are only ever guaranteed sorted by ``primary_key`` (the same
  key ``objects._check_key_order`` enforces at ingest); validating against
  ``orderable_columns`` instead would accept an order the physical files
  cannot actually deliver without an in-memory merge-sort, which §8.2 step 7
  forbids ("never sort an unbounded result in memory").
* a ``TimeInterval``'s column must equal ``contract.observation_time_column``
  even when that column is not in ``contract.filterable_columns`` (task brief
  decision 2) — ``earnings_events.event_date`` is exactly this case.
* fragment pruning (§8.2 step 5) is conservative: it only ever *proves* a
  fragment cannot match (via its partition key, its leading primary-key
  column's bounds, or its time bounds) and never guesses. A predicate on any
  other column, or a fragment missing bounds, always leaves the fragment as a
  candidate — it will not be excluded from the object-open count for a
  narrower reason this module cannot prove sound.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, this package's own ``errors``/``time_formats``, and
``pyarrow`` (for the physical-type -> Arrow-type table shared with
``repository.py``'s scanner) — never ``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa

from engine.v2.contracts.data import (
    DataQuery,
    FragmentRecord,
    KeyPredicate,
    TableContract,
    TimeInterval,
)

from . import time_formats
from .errors import fail

__all__ = [
    "ARROW_TYPES",
    "arrow_type_for",
    "fragment_may_match",
    "null_array_for",
    "order_key",
    "row_matches",
    "validate_query",
]

#: The closed physical-type vocabulary this package's four legacy tables use
#: (``legacy_mapping.ALLOWED_PHYSICAL_TYPES``, mirrored here rather than
#: imported — that constant lives in the legacy-free mapping module, which
#: this module must not depend on).
ARROW_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "timestamp[ns]": pa.timestamp("ns"),
    "timestamp[us]": pa.timestamp("us"),
}


def arrow_type_for(physical_type: str) -> pa.DataType:
    """The Arrow type a contract's ``physical_type`` string maps to."""
    try:
        return ARROW_TYPES[physical_type]
    except KeyError:
        raise fail("CONTRACT_MISMATCH",
                  f"unsupported physical_type {physical_type!r} for a scan") from None


def null_array_for(physical_type: str, length: int) -> pa.Array:
    """A typed-null Arrow array — how a declared-missing nullable column is
    synthesized (§8.2 step 9)."""
    return pa.nulls(length, type=arrow_type_for(physical_type))


# --------------------------------------------------------------------------
# §8.2 step 3: validate columns/predicates/order/limits against the contract
# --------------------------------------------------------------------------


def validate_query(contract: TableContract, query: DataQuery) -> None:
    """Everything ``documents.decode_document`` cannot check because it needs
    the contract: known columns, filterable predicate/interval columns, the
    v1 full-primary-key order, and limits within the contract's own caps.
    """
    declared = {c.name for c in contract.columns}
    _check_columns_known(query.columns, declared)
    _check_limits(contract, query)
    _check_order_by(contract, query)
    for predicate in query.key_filter:
        _check_predicate_column(contract, predicate)
    if query.time_interval is not None:
        _check_time_interval_column(contract, query.time_interval)


def _check_columns_known(columns, declared: set) -> None:
    unknown = [c for c in columns if c not in declared]
    if unknown:
        raise fail("CONTRACT_MISMATCH", f"unknown column(s) {unknown} for this table")


def _check_limits(contract: TableContract, query: DataQuery) -> None:
    if query.max_batch_rows > contract.maximum_batch_rows:
        raise fail("QUERY_NOT_BOUNDED", "max_batch_rows exceeds the table contract's cap")
    if query.max_result_rows > contract.maximum_result_rows:
        raise fail("QUERY_NOT_BOUNDED", "max_result_rows exceeds the table contract's cap")


def _check_order_by(contract: TableContract, query: DataQuery) -> None:
    if tuple(query.order_by) != tuple(contract.primary_key):
        raise fail("CONTRACT_MISMATCH", "order_by must equal the table's full primary key")


def _check_predicate_column(contract: TableContract, predicate: KeyPredicate) -> None:
    if predicate.column not in contract.filterable_columns:
        raise fail("CONTRACT_MISMATCH", f"{predicate.column!r} is not a filterable column")


def _check_time_interval_column(contract: TableContract, interval: TimeInterval) -> None:
    if interval.column != contract.observation_time_column:
        raise fail("CONTRACT_MISMATCH",
                  "a time_interval column must be the table's observation_time_column")


# --------------------------------------------------------------------------
# §8.2 step 5: fragment pruning
# --------------------------------------------------------------------------


def fragment_may_match(record: FragmentRecord, contract: TableContract, query: DataQuery) -> bool:
    """False only when partition/key/time bounds *prove* no row in ``record``
    could satisfy ``query``. Never a false negative."""
    if not _partition_may_match(record, contract, query):
        return False
    if not _leading_key_may_match(record, contract, query):
        return False
    return _time_may_match(record, query)


def _partition_may_match(record: FragmentRecord, contract: TableContract, query: DataQuery) -> bool:
    for predicate in query.key_filter:
        if predicate.column in contract.partition_columns:
            if record.partition_key not in {str(v) for v in predicate.values}:
                return False
    return True


def _leading_key_may_match(record: FragmentRecord, contract: TableContract, query: DataQuery) -> bool:
    if not contract.primary_key or record.primary_key_min is None:
        return True
    leading = contract.primary_key[0]
    physical = _physical_type(contract, leading)
    lo, hi = record.primary_key_min[0], record.primary_key_max[0]
    for predicate in query.key_filter:
        if predicate.column != leading:
            continue
        values = [_comparable_value(v, physical) if not isinstance(v, str) else v
                  for v in predicate.values]
        if all(v < lo or v > hi for v in values):
            return False
    return True


def _time_may_match(record: FragmentRecord, query: DataQuery) -> bool:
    interval = query.time_interval
    if interval is None or record.time_min is None or record.time_max is None:
        return True
    if interval.end_exclusive is not None and record.time_min >= _normalize_bound(interval.end_exclusive):
        return False
    if interval.start_inclusive is not None and record.time_max < _normalize_bound(interval.start_inclusive):
        return False
    return True


# --------------------------------------------------------------------------
# row-level predicate matching and sort-key extraction (used after decoding
# an Arrow batch to native Python values)
# --------------------------------------------------------------------------


def row_matches(row: dict, contract: TableContract, query: DataQuery) -> bool:
    """True iff a decoded row (native python values, keyed by column name)
    satisfies every ``key_filter`` predicate and the ``time_interval``."""
    for predicate in query.key_filter:
        if not _predicate_matches(row, contract, predicate):
            return False
    if query.time_interval is not None and not _interval_matches(row, contract, query.time_interval):
        return False
    return True


def order_key(row: dict, contract: TableContract) -> tuple:
    """The comparable sort-key tuple for ``row``, in ``contract.primary_key``
    order — what the streaming k-way merge across fragments sorts on."""
    return tuple(_comparable_value(row[name], _physical_type(contract, name))
                for name in contract.primary_key)


def _physical_type(contract: TableContract, name: str) -> str:
    return next(c.physical_type for c in contract.columns if c.name == name)


def _comparable_value(value, physical_type: str):
    """A row value in the same comparable form as a naive-timestamp/date
    bound: timestamps become the shared naive-timestamp string; everything
    else (str/int/bool) is already directly comparable."""
    if value is not None and physical_type.startswith("timestamp"):
        return time_formats.format_naive_timestamp(value)
    return value


def _normalize_bound(value: str) -> str:
    """The one comparable form for a ``TimeInterval``/predicate bound --
    the same shared naive-timestamp wire form :func:`_comparable_value`
    produces for a stored row, so the two sides of every comparison are
    always literally the same shape:

    * a bare ``YYYY-MM-DD`` date is midnight of that date;
    * a naive timestamp (already exactly the wire form) is taken as UTC,
      unchanged -- this is the fast, byte-identical path every current
      caller (recorded fragment bounds, ``legacy_materialization``'s bare
      dates and naive timestamps) takes;
    * a timezone-aware timestamp (``Z``, ``+00:00``, or any other offset)
      is converted to UTC before being dropped to the naive wire form.

    ``_time_may_match`` (fragment pruning), ``_predicate_matches``, and
    ``_interval_matches`` (row filtering) all route their bounds through
    this one function, so a boundary can never be included by one path and
    excluded by another. An unparseable bound refuses typed rather than
    being compared as a raw, mismatched string.
    """
    if time_formats.is_naive_timestamp(value):
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise fail("CONTRACT_MISMATCH", "a time bound is not a parseable date or timestamp") from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return time_formats.format_naive_timestamp(parsed)


def _predicate_matches(row: dict, contract: TableContract, predicate: KeyPredicate) -> bool:
    physical = _physical_type(contract, predicate.column)
    value = row.get(predicate.column)
    if value is None:
        return False
    comparable = _comparable_value(value, physical)
    if physical.startswith("timestamp"):
        wanted = {_normalize_bound(v) for v in predicate.values}
    else:
        wanted = set(predicate.values)
    return comparable in wanted


def _interval_matches(row: dict, contract: TableContract, interval: TimeInterval) -> bool:
    physical = _physical_type(contract, interval.column)
    value = row.get(interval.column)
    if value is None:
        return False
    comparable = _comparable_value(value, physical)
    if interval.start_inclusive is not None and comparable < _normalize_bound(interval.start_inclusive):
        return False
    if interval.end_exclusive is not None and comparable >= _normalize_bound(interval.end_exclusive):
        return False
    return True
