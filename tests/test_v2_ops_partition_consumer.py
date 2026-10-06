"""Consumer tests for ``engine.v2.ops.pinned_partition_reader``.

A small synthetic in-memory fake repository serves the reader's whole pinned
scan surface — ``table_contract``/``fragment_records``/``scan_population_bound``
/``scan`` — over ``to_pylist()`` batches and captures every ``DataQuery`` and
scan call. No real dataset, no heavy job, no memory measurement: the retained
claims are about the reader's own accounting, not the OS. The existing
``computed_moves_store._scan_rows`` is the byte-for-byte behavioral reference
for the golden multi-ticker case (R4/parity); the reader must flatten to exactly
what the whole-list reader produced, never silently filtering a ticker, date,
key or null.

Preserved invariants exercised here:

* R1 — a missing/incompatible pinned input keeps the repository's own typed
  refusal and the scan is never reached.
* R2/R5 — a yielded batch is provisional: an output assigned only after full
  exhaustion never sees a partial read, and a mid-stream failure leaves the
  live lease released.
* R3 — a typed integrity refusal propagates as the same exception instance,
  with no retry and no fallback (exactly one scan call).
* R4 — one pinned snapshot, every represented year partition, contract PK order.
* R6 — identical complete inputs reproduce byte-identical serialized output.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.contracts.data import DatasetVersionRef, SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops import computed_moves_store
from engine.v2.ops.pinned_partition_reader import (
    RetainedBatch,
    RetainedRowCount,
    iter_pinned_scan_batches,
)
from tests.data_scan_support import contract_for, contract_ref_for, fake_hash

_TABLE = "daily_market"
_COLUMNS = ("ticker", "date", "implied_move", "year", "revision")

_DM = contract_for(_TABLE)
_DM_REF = contract_ref_for(_DM)


# --------------------------------------------------------------------------
# synthetic fixture: a partition record, the batch/repo fakes, and a snapshot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PartitionRecord:
    """Only ``partition_key`` matters to the reader's year discovery."""

    partition_key: str


class _Batch:
    """A batch returning fresh row dictionaries like PyArrow ``to_pylist()``."""

    def __init__(self, rows) -> None:
        self._rows = rows

    def to_pylist(self):
        return copy.deepcopy(self._rows)


class _PinnedFakeRepository:
    """The ``Repository`` surface ``iter_pinned_scan_batches`` walks, with the
    runtime ``DataQuery``/``KeyPredicate`` contract and ``to_pylist()`` batches.
    Records every query, bound and scan call so a test can prove the scan scope
    and that R1 refusals never reach ``scan``."""

    def __init__(self, snapshot, *, partition_keys, batches, population=None,
                 pre_scan_failure=None, scan_failure=None, scan_failure_after=0,
                 bound_failure=None) -> None:
        self._snapshot = snapshot
        self._records = tuple(_PartitionRecord(key) for key in partition_keys)
        self._batches = [list(rows) for rows in batches]
        self._population = population
        self._pre_scan_failure = pre_scan_failure
        self._scan_failure = scan_failure
        self._scan_failure_after = scan_failure_after
        self._bound_failure = bound_failure
        self.scan_calls = 0
        self.queries = []
        self.bound_calls = []

    @property
    def selected_population(self) -> int:
        if self._population is not None:
            return self._population
        return sum(len(rows) for rows in self._batches)

    def table_contract(self, snapshot_ref, table_name):
        if self._pre_scan_failure is not None:
            raise self._pre_scan_failure
        if table_name not in snapshot_ref.table_versions:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                            details={"table_name": table_name})
        return _DM

    def fragment_records(self, snapshot_ref, table_name):
        return self._records

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref,
                              key_filter=(), time_interval=None) -> int:
        self.bound_calls.append({"snapshot_id": snapshot_id, "table_name": table_name,
                                 "table_contract_ref": table_contract_ref,
                                 "key_filter": key_filter, "time_interval": time_interval})
        if self._bound_failure is not None:
            raise self._bound_failure
        version = self._snapshot.table_versions.get(table_name)
        if version is None or table_contract_ref != version.table_contract_ref:
            raise data_fail("CONTRACT_MISMATCH",
                            "table_contract_ref does not match the pinned version",
                            details={"table_name": table_name})
        return self.selected_population

    def scan(self, query, *, table_name):
        self.scan_calls += 1
        self.queries.append(query)
        for index, rows in enumerate(self._batches):
            if self._scan_failure is not None and index == self._scan_failure_after:
                raise self._scan_failure
            yield _Batch(rows)
        if self._scan_failure is not None and self._scan_failure_after >= len(self._batches):
            raise self._scan_failure


def _snapshot(*, with_table: bool = True) -> SnapshotRef:
    versions = {}
    modes = {}
    if with_table:
        versions[_TABLE] = DatasetVersionRef(dataset_version_id="dsv-partition",
                                             table_contract_ref=_DM_REF,
                                             manifest_hash=fake_hash("partition-manifest"))
        modes[_TABLE] = "reconstructed"
    return SnapshotRef(snapshot_id="snap-partition", manifest_hash=fake_hash("partition-snapshot"),
                       table_versions=versions, calendar_version="cal.v1",
                       source_priority_version="prio.v1", finality_receipt_refs=(),
                       knowledge_mode_by_table=modes)


def _row(ticker, day, year, implied_move, revision):
    return {"ticker": ticker, "date": day, "year": year,
            "implied_move": implied_move, "revision": revision}


# Golden multi-ticker input: null values, a corrected final value for one key,
# and dates sitting on year boundaries (2022-12-31 / 2023-01-01 and
# 2024-12-31 / 2025-01-01). Every key is unique and the rows use PK order.
_GOLDEN_ROWS = [
    _row("AAA", "2022-12-31", 2022, None, 0),
    _row("AAA", "2023-01-01", 2023, 0.02, 0),
    _row("BBB", "2023-06-15", 2023, 0.03, 1),
    _row("CCC", "2024-02-29", 2024, 0.05, 0),
    _row("CCC", "2024-12-31", 2024, None, 0),
    _row("CCC", "2025-01-01", 2025, 0.04, 0),
]
_GOLDEN_YEARS = (2022, 2023, 2024, 2025)


def _flatten(repository, snapshot, *, max_retained_rows, account):
    """Flatten the leased batches in order, reading each lease inside its
    context (the only way to read it) before it is released on the next advance."""
    out: list[dict] = []
    for lease in iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                          max_retained_rows=max_retained_rows,
                                          retained_rows=account):
        with lease as rows:
            out.extend(rows)
    return out


# --------------------------------------------------------------------------
# R4 / parity: the reader flattens to exactly the whole-list reader
# --------------------------------------------------------------------------


def test_golden_batches_match_scan_rows_and_keep_every_population_member():
    snapshot = _snapshot()
    originals = copy.deepcopy(_GOLDEN_ROWS)
    repository = _PinnedFakeRepository(
        snapshot, partition_keys=("2024", "2022", "2025", "2023"),
        batches=[_GOLDEN_ROWS[:2], _GOLDEN_ROWS[2:5], _GOLDEN_ROWS[5:]])
    account = RetainedRowCount()

    flattened = _flatten(repository, snapshot, max_retained_rows=50_000, account=account)
    reference: list[dict] = []
    for lease in computed_moves_store._scan_rows(repository, snapshot, _TABLE, _COLUMNS):
        with lease as batch:
            reference.extend(batch)

    # Byte-for-byte the same rows, in the same order, nothing filtered out.
    assert flattened == reference
    assert flattened == originals
    assert repository.scan_calls == 2  # one per reader, one per reference scan

    # The golden source history is valid: (ticker, date) primary keys are unique.
    assert len({(row["ticker"], row["date"]) for row in _GOLDEN_ROWS}) == len(_GOLDEN_ROWS)

    # No ticker/date/key/null was silently dropped.
    assert [row["ticker"] for row in flattened] == [row["ticker"] for row in originals]
    assert [row["date"] for row in flattened] == [row["date"] for row in originals]
    assert [(row["ticker"], row["date"], row["revision"]) for row in flattened] \
        == [(row["ticker"], row["date"], row["revision"]) for row in originals]
    nulls = [row for row in flattened if row["implied_move"] is None]
    assert len(nulls) == sum(1 for row in originals if row["implied_move"] is None)
    assert {(row["year"], row["date"]) for row in flattened if row["date"] in
            ("2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01")} == {
        (2022, "2022-12-31"), (2023, "2023-01-01"), (2024, "2024-12-31"), (2025, "2025-01-01")}

    # The scan scope: one pinned snapshot, every represented year, PK order.
    reader_query = repository.queries[0]
    assert reader_query.snapshot_id == snapshot.snapshot_id
    assert reader_query.table_contract_ref == snapshot.table_versions[_TABLE].table_contract_ref
    assert reader_query.columns == _COLUMNS
    assert reader_query.order_by == tuple(_DM.primary_key) == ("ticker", "date")
    (predicate,) = reader_query.key_filter
    assert predicate == KeyPredicate(column="year", operator="in", values=_GOLDEN_YEARS)

    # R4: the reader's own query is identical to the whole-list reader's query.
    assert reader_query == repository.queries[1]
    assert account.live_rows == 0


# --------------------------------------------------------------------------
# R6: a history larger than the retained cap still streams whole and exactly
# --------------------------------------------------------------------------


def _large_history(total: int, batch_size: int) -> list[list[dict]]:
    batches, index = [], 0
    while index < total:
        chunk = []
        for offset in range(batch_size):
            if index + offset >= total:
                break
            year = 2020 + (index + offset) % 6
            chunk.append(_row(f"TK{index + offset:03d}", f"{year}-01-01", year, 0.1, 0))
        batches.append(chunk)
        index += batch_size
    return batches


def _serialize(rows: list[dict]) -> bytes:
    return json.dumps(rows, sort_keys=True, default=str).encode("utf-8")


def test_large_history_streams_to_its_end_with_bounded_retention():
    snapshot = _snapshot()
    total, batch_size, retained_cap = 60, 4, 6
    batches = _large_history(total, batch_size)
    repository = _PinnedFakeRepository(snapshot, partition_keys=("2020", "2021", "2022",
                                                                  "2023", "2024", "2025"),
                                       batches=batches)
    account = RetainedRowCount()

    flattened = _flatten(repository, snapshot, max_retained_rows=retained_cap, account=account)
    expected = [row for chunk in batches for row in chunk]

    # Full content, exact and in order, even though the history exceeds the cap.
    assert flattened == expected
    assert len(flattened) == total

    # The result limit IS the selected population (history > retained cap), and
    # the batch cap stays bounded by the retained cap and the contract maximum.
    (query,) = repository.queries
    assert query.max_result_rows == repository.selected_population == total
    assert total > retained_cap
    assert query.max_batch_rows <= retained_cap
    assert query.max_batch_rows <= _DM.maximum_batch_rows

    # Retention is bounded by the cap regardless of how long the history is: at
    # most one in-flight batch (batch_size) is ever live, not the whole 60.
    assert max(len(chunk) for chunk in batches) <= query.max_batch_rows
    assert account.peak_rows == batch_size
    assert account.peak_rows <= retained_cap
    assert account.live_rows == 0

    # R6: two complete identical runs reproduce byte-identical serialized output.
    other = _PinnedFakeRepository(snapshot, partition_keys=("2020", "2021", "2022",
                                                            "2023", "2024", "2025"),
                                   batches=batches)
    second = _flatten(other, snapshot, max_retained_rows=retained_cap, account=RetainedRowCount())
    assert _serialize(flattened) == _serialize(second)


# --------------------------------------------------------------------------
# R1: a missing/incompatible pinned input keeps the repository's typed refusal
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["SNAPSHOT_NOT_FOUND", "CONTRACT_MISMATCH"])
def test_missing_pinned_input_propagates_without_reaching_scan(code):
    problem = data_fail(code, f"synthetic {code}")
    snapshot = _snapshot()
    repository = _PinnedFakeRepository(snapshot, partition_keys=("2024",),
                                       batches=[_GOLDEN_ROWS], pre_scan_failure=problem)
    account = RetainedRowCount()

    with pytest.raises(DataError) as exc:
        list(iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                      max_retained_rows=50_000, retained_rows=account))

    assert exc.value is problem
    assert exc.value.code == code
    assert repository.scan_calls == 0
    assert account.live_rows == 0


def test_no_year_discovery_probes_the_unfiltered_bound_and_empty_population_yields_nothing():
    snapshot = _snapshot()
    problem = data_fail("MANIFEST_CORRUPT", "synthetic unfiltered population-bound failure")
    corrupt = _PinnedFakeRepository(snapshot, partition_keys=(), batches=(),
                                    population=1, bound_failure=problem)
    account = RetainedRowCount()

    with pytest.raises(DataError) as exc:
        list(iter_pinned_scan_batches(corrupt, snapshot, _TABLE, _COLUMNS,
                                      max_retained_rows=50_000, retained_rows=account))

    # R1: the bound's typed refusal propagates as the very same exception ...
    assert exc.value is problem
    assert exc.value.code == "MANIFEST_CORRUPT"
    # ... reached only through the unfiltered no-year probe, never a scan,
    # and nothing is left live.
    (call,) = corrupt.bound_calls
    assert call["key_filter"] == ()
    assert corrupt.scan_calls == 0
    assert account.live_rows == 0

    # A valid empty population (bound 0, no year records) is not a failure: the
    # unfiltered bound is probed exactly once, nothing is scanned, nothing
    # yields.
    empty = _PinnedFakeRepository(snapshot, partition_keys=(), batches=(), population=0)
    leases = list(iter_pinned_scan_batches(empty, snapshot, _TABLE, _COLUMNS,
                                           max_retained_rows=50_000,
                                           retained_rows=RetainedRowCount()))
    assert leases == []
    assert len(empty.bound_calls) == 1
    assert empty.bound_calls[0]["key_filter"] == ()
    assert empty.scan_calls == 0


# --------------------------------------------------------------------------
# R2/R3/R5: a mid-partition integrity refusal is terminal and discards the
# provisional read — nothing published, the live lease released
# --------------------------------------------------------------------------


def test_mid_stream_integrity_failure_discards_provisional_output():
    problem = data_fail("OBJECT_CORRUPT", "synthetic later-fragment integrity failure")
    snapshot = _snapshot()
    repository = _PinnedFakeRepository(
        snapshot, partition_keys=("2022", "2023", "2024", "2025"),
        batches=[_GOLDEN_ROWS[:3], _GOLDEN_ROWS[3:5], _GOLDEN_ROWS[5:]],
        scan_failure=problem, scan_failure_after=1)
    account = RetainedRowCount()

    provisional: list[dict] = []
    first_batch_delivered = None
    published = None  # only ever assigned after the iterator is fully exhausted
    batches_seen = 0
    with pytest.raises(DataError) as exc:
        try:
            for lease in iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                                  max_retained_rows=50_000, retained_rows=account):
                batches_seen += 1
                with lease as rows:
                    assert account.live_rows >= len(rows)
                    if batches_seen == 1:
                        first_batch_delivered = copy.deepcopy(rows)
                    provisional.extend(rows)
            published = list(provisional)
        except DataError:
            provisional.clear()  # discard all attempt-local rows before propagating the refusal
            raise

    assert exc.value is problem
    assert exc.value.problem.category == "integrity"
    assert exc.value.problem.retryable is False
    assert repository.scan_calls == 1
    assert batches_seen == 1
    assert account.live_rows == 0
    # The one batch reached the consumer as provisional input ...
    assert provisional == []
    assert first_batch_delivered == _GOLDEN_ROWS[:3]
    # ... but the failure prevented full exhaustion, so nothing was ever published.
    assert published is None


def test_lease_release_clears_the_exact_list_handed_back_by_the_context():
    snapshot = _snapshot()
    repository = _PinnedFakeRepository(
        snapshot, partition_keys=("2022", "2023", "2024", "2025"),
        batches=[_GOLDEN_ROWS[:2], _GOLDEN_ROWS[2:5], _GOLDEN_ROWS[5:]])
    account = RetainedRowCount()

    held: list[list[dict]] = []
    read_back: list[list[dict]] = []
    for lease in iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                          max_retained_rows=50_000, retained_rows=account):
        assert isinstance(lease, RetainedBatch)
        with lease as rows:
            held.append(rows)  # the exact list object the ``with`` handed back
            read_back.append(list(rows))  # a copy taken while the lease is live
            assert account.live_rows == len(rows)
        # Leaving the context releases the lease: that same list object is now
        # empty, the lease reports itself released, and nothing is left charged.
        assert rows == []
        assert lease.released
        assert account.live_rows == 0

    # Every batch arrived whole and in order while leased ...
    assert read_back == [_GOLDEN_ROWS[:2], _GOLDEN_ROWS[2:5], _GOLDEN_ROWS[5:]]
    # ... and every list retained past its context is emptied by the release.
    assert all(rows == [] for rows in held)
    assert account.live_rows == 0
    assert account.peak_rows == 3


def test_iterator_advance_releases_an_open_lease():
    snapshot = _snapshot()
    repository = _PinnedFakeRepository(
        snapshot, partition_keys=("2022", "2023", "2024", "2025"),
        batches=[_GOLDEN_ROWS[:3], _GOLDEN_ROWS[3:5], _GOLDEN_ROWS[5:]])
    account = RetainedRowCount()

    iterator = iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                        max_retained_rows=50_000, retained_rows=account)
    try:
        first = next(iterator)
        rows = first.__enter__()  # held open: the context is never exited before the advance
        assert rows == _GOLDEN_ROWS[:3]
        assert not first.released
        assert account.live_rows == len(rows)

        second = next(iterator)  # R5: advancing releases the lease the caller moved past
        assert rows == []
        assert first.released
        assert account.live_rows == len(_GOLDEN_ROWS[3:5])
        assert not second.released
    finally:
        iterator.close()  # releases the second lease, so a failure cannot leak one

    assert account.live_rows == 0


def test_iterator_close_releases_an_open_lease():
    snapshot = _snapshot()
    repository = _PinnedFakeRepository(
        snapshot, partition_keys=("2022", "2023", "2024", "2025"),
        batches=[_GOLDEN_ROWS[:3], _GOLDEN_ROWS[3:5], _GOLDEN_ROWS[5:]])
    account = RetainedRowCount()

    iterator = iter_pinned_scan_batches(repository, snapshot, _TABLE, _COLUMNS,
                                        max_retained_rows=50_000, retained_rows=account)
    lease = next(iterator)
    rows = lease.__enter__()  # held open: there is no context block to exit
    try:
        assert rows == _GOLDEN_ROWS[:3]
        assert account.live_rows == len(rows)
    finally:
        iterator.close()  # R5: iterator close releases the open lease
    assert rows == []
    assert lease.released
    assert account.live_rows == 0


@pytest.mark.parametrize("mutation", ["pop", "append"])
def test_release_discharges_the_creation_charge_after_the_list_is_edited(mutation):
    account = RetainedRowCount()
    lease = RetainedBatch([_row("AAA", "2024-01-01", 2024, 0.01, 0),
                           _row("BBB", "2024-01-02", 2024, 0.02, 0)], account)
    rows = []
    try:
        rows = lease.__enter__()  # the exact mutable list the lease holds
        assert account.live_rows == 2
        if mutation == "pop":
            rows.pop()
        else:
            rows.append(_row("CCC", "2024-01-03", 2024, 0.03, 0))
        # The charge was fixed at creation: caller edits move neither it nor the
        # account until the release.
        assert account.live_rows == 2
    finally:
        lease.release()  # cleanup, so a failure never leaves a live lease
    assert rows == []
    assert lease.released
    assert account.live_rows == 0
    assert account.peak_rows == 2
