"""The shared pinned-partition reader: one bounded scan, streamed as leased
batches (ops ARCHITECTURE.md "Inputs"). Slice 1 is additive only — both stores'
``_scan_rows`` and their callers keep their current behavior until their own
separately scoped migrations.

Why a reader instead of a list: ``_scan_rows`` materializes a whole selection
into one Python list, so the peak retained input equals the entire history. This
keeps exactly that query scope — one pinned snapshot, every represented year
partition, the table contract's primary-key order, no ticker or date filter, no
change to null/correction/date semantics — and hands back one batch at a time.
Each batch is held by a :class:`RetainedBatch` lease accounted in
:class:`RetainedRowCount`, so what is retained at once is bounded by
``max_retained_rows`` while the history behind it is not.

**Yielded batches are provisional until the iterator is exhausted.** A scan can
raise after the first batch (``Repository._yield_batches`` re-checks strict
primary-key order per row, and a later fragment's bytes can fail integrity), so
a caller that publishes after a partial read publishes provisional input.
Publish only after complete exhaustion and validation of every selected
partition; otherwise discard the attempt. The reader commits and publishes
nothing itself.

Preserved invariants (R1-R6):

* R1 — missing, corrupt or incompatible pinned input keeps the repository's own
  typed refusal (``CONTRACT_MISMATCH``, ``MANIFEST_CORRUPT``,
  ``QUERY_NOT_BOUNDED``); never an empty result, a newer snapshot or another
  source.
* R2 — output remains provisional until every selected partition is exhausted
  and validated, as above.
* R3 — integrity refusals are terminal: every repository exception propagates
  unchanged, with no retry and no fallback here.
* R4 — one pinned snapshot and one scope: a single ``DataQuery`` over
  ``snapshot.snapshot_id`` and that snapshot's own contract ref.
* R5 — a mid-partition failure discards provisional state: nothing is retained
  across batches but the counters, the live lease is released on scan error and
  on iterator close, and nothing is published.
* R6 — identical complete inputs yield the same ordered batches, so a caller
  serializing them reproduces byte-identical output.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef, TableContract
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail

__all__ = ["RetainedBatch", "RetainedRowCount", "iter_pinned_scan_batches"]

#: The batch ceiling both stores' ``_scan_rows`` already apply on top of the
#: table contract's own maximum; the retained cap is the caller's, not this one.
_MAX_BATCH_ROWS = 50_000

#: The year-partition key column ``_scan_rows`` selects every represented
#: partition of, sorted and unique, so the query scope matches it exactly.
_PARTITION_COLUMN = "year"


@dataclass
class RetainedRowCount:
    """Simultaneously retained input rows: the live count and its all-time peak.

    One account per read, owned by the caller, so ``peak_rows`` is the number it
    must plan against (the existing ``MAX_SCAN_ROWS`` values stay the ceiling).
    """

    live_rows: int = 0
    peak_rows: int = 0

    def retain(self, rows: int) -> None:
        """Charge ``rows`` as live and raise the peak if the account is now higher."""
        self.live_rows += rows
        if self.live_rows > self.peak_rows:
            self.peak_rows = self.live_rows

    def discharge(self, rows: int) -> None:
        """Return ``rows``: a lease gives back exactly what it took."""
        self.live_rows -= rows


class RetainedBatch:
    """One scanned batch, held under a lease on :class:`RetainedRowCount`.

    ``with lease as rows:`` is the only way to read the batch, and leaving the
    block releases it: the live count drops and the row list is cleared, so
    released rows cannot be read again — the caller processes a lease before
    advancing, and advancing releases it.
    """

    def __init__(self, rows: list[dict[str, Any]], retained_rows: RetainedRowCount) -> None:
        self._rows = rows
        self._retained_rows = retained_rows
        self._released = False
        retained_rows.retain(len(rows))

    @property
    def released(self) -> bool:
        return self._released

    def __enter__(self) -> list[dict[str, Any]]:
        if self._released:
            raise fail("INVALID_REQUEST", "released retained batch lease re-entered")
        return self._rows

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        self.release()
        return False

    def release(self) -> None:
        """Idempotent release, so a context exit and an advance cannot double-pay."""
        if self._released:
            return
        self._released = True
        self._retained_rows.discharge(len(self._rows))
        self._rows = []


def _bounded_limits(contract: TableContract, population_bound: int,
                    max_retained_rows: int) -> tuple[int, int]:
    """``(max_batch_rows, max_result_rows)`` for one pinned selection.

    ``max_result_rows`` IS the manifest population bound, never the retained cap:
    a history larger than ``max_retained_rows`` still streams to its end, it just
    never arrives all at once. ``max_batch_rows`` is what one lease can hold, so
    it is clamped by the contract maximum, by 50,000 and by the retained cap, and
    by the population when positive — that last clamp is what keeps the batch
    limit at or below the result limit, which ``documents`` otherwise refuses. A
    zero population is a valid empty selection: the result limit stays 0 while
    the batch limit keeps a positive value, as the existing readers do.
    """
    max_batch_rows = min(contract.maximum_batch_rows, _MAX_BATCH_ROWS, max_retained_rows)
    if population_bound > 0:
        max_batch_rows = min(max_batch_rows, population_bound)
    return max_batch_rows, population_bound


def iter_pinned_scan_batches(repository: Repository, snapshot: SnapshotRef,
                             table_name: str, columns: Sequence[str], *,
                             max_retained_rows: int,
                             retained_rows: RetainedRowCount) -> Iterator[RetainedBatch]:
    """Stream one pinned snapshot's ``table_name`` as leased batches in key order.

    Yields one :class:`RetainedBatch` per ``repository.scan`` batch, after
    ``to_pylist()``, in the same order ``_scan_rows`` materialized them. Every
    repository exception propagates unchanged (R1/R3); an empty partition
    discovery yields nothing at all. See the module docstring for why a yielded
    batch is provisional until exhaustion.
    """
    if (isinstance(max_retained_rows, bool) or not isinstance(max_retained_rows, int)
            or max_retained_rows <= 0):
        raise fail("INVALID_REQUEST",
                   "pinned partition max_retained_rows must be a positive integer")
    # The contract lookup is first because it is the typed refusal (R1) for a
    # table this snapshot does not pin; the dict walk below only reads what it
    # already accepted.
    contract = repository.table_contract(snapshot, table_name)
    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    if not years:
        return
    key_filter = (KeyPredicate(column=_PARTITION_COLUMN, operator="in", values=years),)
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=key_filter, time_interval=None)
    max_batch_rows, max_result_rows = _bounded_limits(
        contract, population_bound, max_retained_rows)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=tuple(columns),
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    lease: RetainedBatch | None = None
    try:
        for batch in repository.scan(query, table_name=table_name):
            if lease is not None:
                lease.release()  # advancing releases the batch the caller moved past
            lease = RetainedBatch(batch.to_pylist(), retained_rows)
            yield lease
    finally:
        # Exhaustion, a scan error and iterator close all land here (R5): the
        # reader never leaves a lease charged to the account.
        if lease is not None:
            lease.release()
