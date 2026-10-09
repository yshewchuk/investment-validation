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
``engine.v2.contracts``, ``engine.v2.foundation``'s document helpers, this
package's own ``documents``/``errors``/``time_formats``, and ``pyarrow``
(for the physical-type -> Arrow-type table shared with
``repository.py``'s scanner) — never ``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc

from engine.v2.contracts.data import (
    DataQuery,
    FragmentRecord,
    KeyPredicate,
    TableContract,
    TimeInterval,
)
from engine.v2.foundation import DocumentError, to_document

from . import documents, time_formats
from .errors import fail

__all__ = [
    "ARROW_TYPES",
    "ScanPopulation",
    "arrow_type_for",
    "compile_batch_matcher",
    "compile_row_matcher",
    "fragment_may_match",
    "null_array_for",
    "order_key",
    "plan_scan_population",
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
    v1 full-primary-key order, and the batch limit within the contract's own
    cap. A table states no result-row cap, so ``max_result_rows`` is the
    query's own bound and is not compared against the contract here.
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
# §8.2 step 5: fragment pruning and membership-bound planning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanPopulation:
    """The candidate fragment set one selection opens, and the manifest row
    count they carry in total. The bound counts candidate rows — never
    matching output rows, and never a memory estimate."""

    records: tuple[FragmentRecord, ...]
    row_count: int


def plan_scan_population(contract: TableContract, records, *,
                         key_filter=(), time_interval=None) -> ScanPopulation:
    """The one membership-planning path behind scans, explains, and
    ``Repository.scan_population_bound``: prune with the exact partition/
    leading-primary-key/time bounds the scan itself uses, then sum the
    surviving fragments' recorded ``row_count`` (Python integers). Selection
    predicates are strictly re-decoded (a malformed one is
    ``QUERY_NOT_BOUNDED``, like a malformed ``DataQuery``), and a surviving
    fragment with non-int or negative metadata is ``MANIFEST_CORRUPT`` before
    any object is opened. No predicates is legal here — a whole-table bound
    is metadata planning, not a scan, which still refuses unbounded queries
    in ``documents._check_data_query``. Error messages carry no path or row
    value (guide §7.2 redaction)."""
    predicates = _decoded_predicates(key_filter)
    interval = _decoded_interval(time_interval)
    for predicate in predicates:
        _check_predicate_column(contract, predicate)
    if interval is not None:
        _check_time_interval_column(contract, interval)
    _check_planning_timestamp_values(contract, predicates, interval)
    surviving = tuple(record for record in records
                      if _fragment_may_match(record, contract, predicates, interval))
    row_count = 0
    for record in surviving:
        count = record.row_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise fail("MANIFEST_CORRUPT",
                       "a surviving fragment's recorded row_count is not valid metadata",
                       details={"fragment_id": record.fragment_id})
        row_count += count
    return ScanPopulation(records=surviving, row_count=row_count)


def _decoded_predicates(key_filter) -> tuple[KeyPredicate, ...]:
    if not isinstance(key_filter, (list, tuple)):
        raise fail("QUERY_NOT_BOUNDED", "key_filter must be a list or tuple of predicates")
    decoded = []
    for predicate in key_filter:
        try:
            decoded.append(documents.decode_document(KeyPredicate, to_document(predicate)))
        except DocumentError as exc:
            raise fail("QUERY_NOT_BOUNDED",
                       f"a key_filter predicate is refused: {exc.code}") from exc
    columns = [predicate.column for predicate in decoded]
    if len(set(columns)) != len(columns):
        raise fail("QUERY_NOT_BOUNDED", "the same column is filtered twice")
    return tuple(decoded)


def _decoded_interval(time_interval) -> TimeInterval | None:
    if time_interval is None:
        return None
    try:
        return documents.decode_document(TimeInterval, to_document(time_interval))
    except DocumentError as exc:
        raise fail("QUERY_NOT_BOUNDED", f"a time_interval is refused: {exc.code}") from exc


def _check_planning_timestamp_values(contract: TableContract, predicates,
                                     interval: TimeInterval | None) -> None:
    """Planning-path timestamp-value validation; same static refusal as the row path."""
    for predicate in predicates:
        if not _physical_type(contract, predicate.column).startswith("timestamp"):
            continue
        for value in predicate.values:
            if not isinstance(value, str):
                raise fail("CONTRACT_MISMATCH",
                           "key_filter values on a timestamp column must be strings")
            _normalize_bound(value)
    if interval is not None:
        for bound in (interval.start_inclusive, interval.end_exclusive):
            if bound is not None:
                _normalize_bound(bound)


def fragment_may_match(record: FragmentRecord, contract: TableContract, query: DataQuery) -> bool:
    """False only when partition/key/time bounds *prove* no row in ``record``
    could satisfy ``query``. Never a false negative. The scan and explain
    paths reach this through :func:`plan_scan_population`; it stays public
    for its unchanged direct callers."""
    return _fragment_may_match(record, contract, query.key_filter, query.time_interval)


def _fragment_may_match(record: FragmentRecord, contract: TableContract, key_filter,
                        interval: TimeInterval | None) -> bool:
    if not _partition_may_match(record, contract, key_filter):
        return False
    if not _leading_key_may_match(record, contract, key_filter):
        return False
    return _time_may_match(record, interval)


def _partition_may_match(record: FragmentRecord, contract: TableContract, key_filter) -> bool:
    for predicate in key_filter:
        if predicate.column in contract.partition_columns:
            if record.partition_key not in {str(v) for v in predicate.values}:
                return False
    return True


def _leading_key_may_match(record: FragmentRecord, contract: TableContract, key_filter) -> bool:
    if not contract.primary_key or record.primary_key_min is None:
        return True
    leading = contract.primary_key[0]
    physical = _physical_type(contract, leading)
    lo, hi = record.primary_key_min[0], record.primary_key_max[0]
    for predicate in key_filter:
        if predicate.column != leading:
            continue
        if physical.startswith("timestamp") and any(not isinstance(value, str)
                                                    for value in predicate.values):
            raise fail("CONTRACT_MISMATCH",
                       "key_filter values are incompatible with fragment key bounds")
        try:
            values = [_comparable_value(v, physical) if not isinstance(v, str) else v
                      for v in predicate.values]
            if all(v < lo or v > hi for v in values):
                return False
        except TypeError:
            raise fail("CONTRACT_MISMATCH",
                       "key_filter values are incompatible with fragment key bounds") from None
    return True


def _time_may_match(record: FragmentRecord, interval: TimeInterval | None) -> bool:
    if interval is None or record.time_min is None or record.time_max is None:
        return True
    if interval.end_exclusive is not None and _normalize_bound(record.time_min) >= _normalize_bound(interval.end_exclusive):
        return False
    if interval.start_inclusive is not None and _normalize_bound(record.time_max) < _normalize_bound(interval.start_inclusive):
        return False
    return True


# --------------------------------------------------------------------------
# row-level predicate matching and sort-key extraction (used after decoding
# an Arrow batch to native Python values)
# --------------------------------------------------------------------------


def row_matches(row: dict, contract: TableContract, query: DataQuery) -> bool:
    """True iff a decoded row (native python values, keyed by column name)
    satisfies every ``key_filter`` predicate and the ``time_interval``."""
    return compile_row_matcher(contract, query)(row)


def compile_row_matcher(contract: TableContract, query: DataQuery) -> Callable[[dict], bool]:
    """A per-query-compiled row predicate: equivalent to calling
    ``row_matches(row, contract, query)`` for every row, but normalizes each
    key_filter predicate's ``values`` and the time_interval's bounds ONCE
    here rather than once per row. Byte-identical semantics to
    ``row_matches`` -- same None handling, same ``_normalize_bound``/
    ``_comparable_value`` routing, same refusal on an unparseable bound.
    Built once per scan (``repository.py`` calls this once, not per row).
    """
    predicates = []
    for predicate in query.key_filter:
        physical = _physical_type(contract, predicate.column)
        if physical.startswith("timestamp"):
            wanted = {_normalize_bound(v) for v in predicate.values}
        else:
            wanted = set(predicate.values)
        predicates.append((predicate.column, physical, wanted))
    interval = None
    if query.time_interval is not None:
        interval_physical = _physical_type(contract, query.time_interval.column)
        interval_start = (None if query.time_interval.start_inclusive is None
                          else _normalize_bound(query.time_interval.start_inclusive))
        interval_end = (None if query.time_interval.end_exclusive is None
                        else _normalize_bound(query.time_interval.end_exclusive))
        interval = (query.time_interval.column, interval_physical, interval_start, interval_end)

    def _matches(row: dict) -> bool:
        for column, physical, wanted in predicates:
            value = row.get(column)
            if value is None:
                return False
            if _comparable_value(value, physical) not in wanted:
                return False
        if interval is not None:
            column, physical, start_norm, end_norm = interval
            value = row.get(column)
            if value is None:
                return False
            comparable = _normalize_bound(value) if physical == "string" else _comparable_value(value, physical)
            if start_norm is not None and comparable < start_norm:
                return False
            if end_norm is not None and comparable >= end_norm:
                return False
        return True

    return _matches


def compile_batch_matcher(contract: TableContract, query: DataQuery
                          ) -> Callable[[dict[str, pa.Array], int], pa.Array] | None:
    """The vectorized form of :func:`compile_row_matcher`, narrowed (task
    brief #286 follow-up) to ``key_filter`` equality/set-membership
    (pyarrow ``compute.is_in``) on the column types this package can prove
    byte-identical to the row path: ``string``, ``int64``, and a
    timestamp column (floored/widened to microsecond resolution exactly
    like the row path's own wire form -- see :func:`_batch_comparable`).
    Returns ``None``, falling the WHOLE query back to
    :func:`compile_row_matcher`, whenever:

    * ``query.time_interval`` is set at all, even with neither bound set
      (a range comparison needs the same bound-parsing the row path's
      ``_normalize_bound``/``_comparable_value`` already carry, and a
      time column declared a non-timestamp physical type -- e.g. a
      string-typed ``observation_time_column`` -- makes even a range
      comparison's own meaning, chronological vs. lexical, a per-column
      decision this function does not make);
    * any ``key_filter`` predicate targets a ``bool`` or ``float64``
      column -- ``pyarrow.compute.is_in`` compares a float's raw bit
      pattern, so ``-0.0`` never matches a ``0`` value_set entry even
      though the row path's plain Python ``==``/set membership treats
      them equal (confirmed directly: a real divergence, not a
      theoretical one);
    * a predicate's ``values`` cannot be represented in its column's
      declared Arrow type, or ANY other exception is raised while
      compiling a predicate for this query -- a future failure mode this
      package has not enumerated degrades to the row path instead of
      crashing the scan.

    This is a compile-time decision from ``contract``/``query`` alone,
    never from the data a caller later feeds the returned closure. A
    caller that gets ``None`` back must fall back to
    ``compile_row_matcher`` tested against every decoded row. A caller
    must ALSO fall back per fragment if the returned closure itself
    raises when called against real batch data
    (``repository.Repository._fragment_rows`` does this): this function's
    own refusal covers only what it can decide from ``contract``/``query``
    alone, not every way real Arrow data could defeat ``is_in`` at
    evaluation time.

    The returned closure takes ``columns`` (every ``key_filter`` column
    name mapped to that column's full Arrow array for one batch -- a
    caller supplies a typed-null array, e.g. :func:`null_array_for`, for a
    column absent from the physical fragment) and ``num_rows`` (the
    batch's row count, used only when there is nothing to filter), and
    returns a boolean mask with no nulls: True iff the row at that index
    satisfies every predicate.
    """
    if query.time_interval is not None:
        return None
    try:
        compiled = [_compile_batch_predicate(contract, p) for p in query.key_filter]
    except Exception:
        return None
    if any(c is None for c in compiled):
        return None

    def _mask(columns: dict[str, pa.Array], num_rows: int) -> pa.Array:
        result = pa.array([True] * num_rows, type=pa.bool_())
        for column, apply_predicate in compiled:
            result = pc.and_(result, apply_predicate(columns[column]))
        return result

    return _mask


_VECTORIZABLE_PHYSICAL_TYPES = frozenset({"string", "int64", "timestamp[ns]", "timestamp[us]"})


def _compile_batch_predicate(contract: TableContract, predicate: KeyPredicate
                             ) -> tuple[str, Callable[[pa.Array], pa.Array]] | None:
    """Compiles one ``key_filter`` predicate for the vectorized mask, or
    returns ``None`` when its column's physical type is not one this
    package can prove byte-identical between the two paths --
    :func:`compile_batch_matcher` treats that exactly like an
    unrepresentable predicate value: fall back the WHOLE query to the row
    path. ``bool``/``float64`` are excluded even though
    ``pyarrow.compute.is_in`` can run on them: a float's bit-pattern
    comparison makes ``-0.0`` never match a ``0`` value_set entry, unlike
    the row path's plain Python equality (task brief #286 follow-up)."""
    physical = _physical_type(contract, predicate.column)
    if physical not in _VECTORIZABLE_PHYSICAL_TYPES:
        return None
    value_set = _batch_value_set(physical, predicate.values)

    def _apply(array: pa.Array) -> pa.Array:
        # ``skip_nulls=True`` is already pyarrow's own behavior for a null
        # input to ``is_in`` (it compares unequal to every concrete
        # value_set member) -- passed explicitly so this never silently
        # starts matching nulls on a pyarrow upgrade.
        return pc.is_in(_batch_comparable(array, physical), value_set=value_set,
                        skip_nulls=True)

    return predicate.column, _apply


def _batch_comparable(array: pa.Array, physical_type: str) -> pa.Array:
    """The vectorized form of :func:`_comparable_value`: a ``timestamp[ns]``
    column is floored to the shared wire form's microsecond resolution and
    widened to ``timestamp[us]`` -- byte-identical to
    ``value.strftime(time_formats.NAIVE_TIMESTAMP_FORMAT)`` on one decoded
    row for every instant, including a pre-1970 (negative-epoch) one.
    Delegates to :func:`_floor_ns_to_us`, which does this with exact
    ``int64`` Arrow compute -- never ``pc.floor_temporal`` (a real pyarrow
    bug near the minimum representable ``timestamp[ns]`` instant: confirmed
    directly to wrap such a value around to the MAXIMUM representable
    instant instead of flooring it) and never a numpy round-trip
    (``to_numpy()`` promotes to ``float64`` once a null is present, which
    cannot hold a full ``int64`` nanosecond tick count exactly). A
    ``timestamp[us]`` column, or ``string``/``int64``, needs no transform:
    it is already directly comparable, same as the row path."""
    if physical_type == "timestamp[ns]":
        return _floor_ns_to_us(array)
    return array


def _floor_ns_to_us(array: pa.Array) -> pa.Array:
    """Floor a ``timestamp[ns]`` array to microsecond resolution and widen
    it to ``timestamp[us]``, entirely in ``int64`` Arrow compute (see
    :func:`_batch_comparable` for why). ``pc.divide`` truncates toward
    zero; the ``needs_floor_adjust`` step corrects that to a true floor
    (round toward negative infinity) exactly when the raw nanosecond tick
    count is negative and the division had a non-zero remainder."""
    ns_int = pc.cast(array, pa.int64())
    truncated = pc.divide(ns_int, 1000)
    remainder = pc.subtract(ns_int, pc.multiply(truncated, 1000))
    needs_floor_adjust = pc.and_(pc.less(ns_int, 0), pc.not_equal(remainder, 0))
    floored_us = pc.if_else(needs_floor_adjust, pc.subtract(truncated, 1), truncated)
    return pc.cast(floored_us, pa.timestamp("us"))


def _batch_value_set(physical_type: str, values) -> pa.Array:
    """The ``value_set`` :func:`_compile_batch_predicate` feeds ``is_in``,
    in the same comparable form :func:`_batch_comparable` casts a column
    to. May raise during compilation -- a pyarrow ``ArrowException`` when a
    value cannot be represented in the column's declared Arrow type, a
    plain ``OverflowError`` for an ``int64`` column given a Python ``int``
    outside the C ``long`` range, or anything else -- caught broadly by
    :func:`compile_batch_matcher`, never by this function."""
    if physical_type.startswith("timestamp"):
        return pa.array([_parsed_bound(v) for v in values], type=pa.timestamp("us"))
    return pa.array(values, type=ARROW_TYPES[physical_type])


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


def _parsed_bound(value: str) -> datetime:
    """The parsed ``datetime`` behind :func:`_normalize_bound`, before it
    is formatted to the shared wire string. The vectorized batch path's
    timestamp ``key_filter`` support (:func:`_batch_value_set`) calls this
    directly instead of calling ``_normalize_bound`` and then reparsing its
    formatted string with ``strptime`` -- that round trip is not always
    safe: ``strftime``'s ``%Y`` does not reliably zero-pad a year below
    1000 on every platform, so a bound like ``"0001-01-01"`` can normalize
    to an unpadded wire string that a 4-digit-year ``strptime`` reparse
    then refuses with a ``ValueError`` -- a real, platform-dependent
    failure this avoids entirely by never formatting to a string and
    reparsing it in the first place. Same ``CONTRACT_MISMATCH`` refusal on
    an unparseable bound as :func:`_normalize_bound`."""
    if time_formats.is_naive_timestamp(value):
        return datetime.strptime(value, time_formats.NAIVE_TIMESTAMP_FORMAT)
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise fail("CONTRACT_MISMATCH", "a time bound is not a parseable date or timestamp") from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


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

    ``_time_may_match`` (fragment pruning) and ``compile_row_matcher`` (row
    filtering) both route their bounds through
    this one function, so a boundary can never be included by one path and
    excluded by another. An unparseable bound refuses typed rather than
    being compared as a raw, mismatched string.
    """
    if time_formats.is_naive_timestamp(value):
        return value
    return time_formats.format_naive_timestamp(_parsed_bound(value))
