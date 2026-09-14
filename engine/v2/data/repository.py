"""Exact snapshot resolution — phase-2 guide §8.1.

``Repository.resolve(snapshot_id)`` rebuilds a ``SnapshotRef`` entirely from
catalog rows, never from ``data_snapshot_heads``: it loads the snapshot row,
every table's dataset version, and every fragment's full ordered membership,
then feeds them back through the *same public builders* that mint identity at
commit time (``manifests.fragment_record``, ``.dataset_manifest``,
``.snapshot_ref`` — ``engine/v2/data/catalog.py`` uses the same three), and
compares the freshly recomputed id/manifest_hash against the catalog's own
primary key at each level. A mismatch — corrupt or missing membership, an
edited row, a dropped join partner — surfaces as ``MANIFEST_CORRUPT``, this
package's judgement call (task brief decision 1: §11 has no
``INTEGRITY_FAILED``, the guide prose's name for the same case). An unknown
``snapshot_id`` is ``SNAPSHOT_NOT_FOUND``.

Every ``FragmentRecord`` field the v1 catalog schema could not store on its
own — only ``input_receipt_refs`` — was added as v2 (task brief decision 3,
``engine/v2/data/schema.py``); every other field (object/contract refs via a
join, partition_key, row_count, byte/logical hashes, key/time bounds,
import_request_hash) was already a v1 column, so no ``manifest_hash`` column
is needed on ``data_fragments`` either: rebuilding through
``manifests.fragment_record`` recomputes it deterministically from the other
stored fields, the same way the original commit did.

One read-only SQLite transaction (:func:`_read_only`) covers the whole walk,
so a concurrent commit elsewhere can never hand back a torn snapshot, and a
resolved ``SnapshotRef`` never changes underneath a caller holding it — every
row this reads is append-only.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and this package's own ``manifests``/``objects``/
``errors`` — never ``engine.v2.ops``.

--------------------------------------------------------------------------
P2-4: bounded scans, exact event/chain lookup, dependency plans
--------------------------------------------------------------------------

``scan``/``get_event``/``get_chain``/``explain_dependencies`` are this
slice's additions (phase-2 guide §5.5, §8.2, §8.3). Pure validation, fragment
pruning, and row-predicate matching live in :mod:`engine.v2.data.query`; pure
legacy-identity mapping lives in :mod:`engine.v2.data.events` and
:mod:`engine.v2.data.chains`. This module stays the one place that touches
SQLite and, now, the object store — every other new module is pure.

Judgement calls (task brief decision 1, P2-4):

* ``Repository.__init__`` takes an optional ``store: ArtifactStore | None``,
  kept keyword-optional so the existing ``Repository(conn)`` call sites
  (``resolve`` never touches object bytes) keep working unchanged; ``scan``
  raises immediately if ``store`` is absent.
* "Implicit latest" (``snapshot_id`` values like ``"latest"``/``"current"``/a
  head-scope name) needs no special rejection code: ``resolve`` never reads
  ``data_snapshot_heads``, so such a value is simply an unknown
  ``snapshot_id`` and already gets ``SNAPSHOT_NOT_FOUND`` from the existing
  code below.
* fragments are read in ``manifests.py``'s own membership order (ascending
  ``partition_key``, and — per the ``data_version_fragments`` ``ordinal``
  column — ascending within a partition too, whatever a partition's fragment
  count) and merged into one global order via a streaming ``heapq.merge`` on
  each row's full primary key, not a sort of the whole result: this is
  correct whether a logical partition holds one fragment or several, and it
  is what makes §8.2 step 7's "process fragments in manifest key-range
  order; never sort an unbounded result in memory" hold even when a query's
  primary key does not happen to be partition-aligned (e.g. a table
  partitioned by year but keyed leading-column by ticker). The merge only
  ever buffers one pending row per still-open fragment.
* the merged stream's own strictly-increasing, unique order is verified end
  to end (across fragment boundaries, not only within one) before each row
  is even appended to a pending batch — the literal implementation of §8.2
  step 7's "include hidden primary-key columns ... to verify ordering" and
  step 9's ordering guarantee. A duplicate key is exactly as much an
  integrity defect as a regression (task 2 review fix), so both raise
  ``MANIFEST_CORRUPT`` here, before the offending row is ever yielded — a
  duplicate ``event_id`` therefore already surfaces as ``MANIFEST_CORRUPT``
  by the time ``events.get_event`` sees its rows, through this path rather
  than a second check of its own.
* the stat-tuple verification cache §8.2 step 6 describes is DEFERRED
  (tech debt TD-1, task brief): every object open re-hashes via
  ``objects.verify_object_path``, unconditionally.
* "deadline exceeded" is its own registered code, ``DEADLINE_EXCEEDED``
  (task 2 review fix) — resource category, retryable (a caller may simply
  retry with a later deadline).
"""
from __future__ import annotations

import heapq
import json
import sqlite3
from contextlib import contextmanager

import pyarrow as pa
import pyarrow.parquet as pq

# Every name below comes from one of exactly eight distinct modules — the
# §4.3 fan-out budget (8, non-orchestrator; ``engine.v2.data`` is not one).
# The scan/decode generators below carry no ``-> Iterator[...]`` return
# annotation for the same reason: a ninth import just to satisfy a type hint
# ruff would still resolve (``from __future__ import annotations`` postpones
# *evaluation*, not ruff's own undefined-name check) is not worth the budget.
# ``errors.fail``/``objects.FragmentInspection`` are reached through the
# already-imported ``errors``/``objects`` modules rather than their own
# separate import lines, and the deadline check uses
# ``foundation.SystemClock`` instead of a bare ``datetime`` import, for the
# same reason.
from engine.v2.contracts.data import (
    CHAIN_QUERY_V1,
    DATA_QUERY_V1,
    ChainQuery,
    ChainSnapshot,
    DataQuery,
    DependencyEntry,
    DependencyPlan,
    EarningsEvent,
    EventRef,
    FragmentRecord,
    ObjectRef,
    SnapshotRef,
    TableContract,
    TableContractRef,
)
from engine.v2.data import chains, documents, errors, events, manifests, objects
from engine.v2.data import query as query_mod
from engine.v2.foundation import (
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    parse_timestamp,
    to_document,
)

__all__ = ["Repository"]


@contextmanager
def _read_only(conn: sqlite3.Connection):
    if conn.in_transaction:
        raise RuntimeError("nested data catalog transaction")
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("ROLLBACK")


class Repository:
    """Read-only access to one already-migrated data catalog connection."""

    def __init__(self, conn: sqlite3.Connection, store: ArtifactStore | None = None) -> None:
        self._conn = conn
        self._store = store

    def resolve(self, snapshot_id: str) -> SnapshotRef:
        with _read_only(self._conn) as conn:
            snap_row = conn.execute(
                "SELECT * FROM data_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
            if snap_row is None:
                raise errors.fail("SNAPSHOT_NOT_FOUND", "unknown snapshot id",
                          details={"snapshot_id": snapshot_id})
            table_rows = conn.execute(
                "SELECT table_name, dataset_version_id FROM data_snapshot_tables "
                "WHERE snapshot_id = ?", (snapshot_id,)).fetchall()
            table_versions = {row["table_name"]: self._manifest(conn, row["dataset_version_id"])
                              for row in table_rows}
            snap = manifests.snapshot_ref(
                table_versions, calendar_version=snap_row["calendar_version"],
                source_priority_version=snap_row["source_priority_version"],
                finality_receipt_refs=tuple(json.loads(snap_row["finality_receipt_refs_json"])),
                parent_snapshot_id=snap_row["parent_snapshot_id"])
            if (snap.snapshot_id != snap_row["snapshot_id"]
                    or snap.manifest_hash != snap_row["manifest_hash"]):
                raise errors.fail("MANIFEST_CORRUPT", "snapshot does not match its catalog identity",
                          details={"snapshot_id": snapshot_id})
            return snap

    def _manifest(self, conn: sqlite3.Connection, dataset_version_id: str):
        dsv_row = conn.execute(
            "SELECT * FROM data_dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,)).fetchone()
        if dsv_row is None:
            raise errors.fail("MANIFEST_CORRUPT", "snapshot references an unknown dataset version",
                      details={"dataset_version_id": dataset_version_id})
        contract_ref = self._contract_ref(conn, dsv_row["contract_id"])
        records = self._records(conn, dataset_version_id, contract_ref)
        evidence = json.loads(dsv_row["evidence_json"])
        partition_hashes = json.loads(dsv_row["partition_logical_hashes_json"])
        manifest = manifests.dataset_manifest(
            contract_ref, records, knowledge_mode=dsv_row["knowledge_mode"],
            coverage_receipt_refs=tuple(evidence["coverage_receipt_refs"]),
            availability_evidence_refs=tuple(evidence["availability_evidence_refs"]),
            parent_dataset_version_id=dsv_row["parent_dataset_version_id"],
            partition_logical_hashes=partition_hashes)
        ref = manifest.dataset_version_ref
        if (ref.dataset_version_id != dsv_row["dataset_version_id"]
                or ref.manifest_hash != dsv_row["manifest_hash"]):
            raise errors.fail("MANIFEST_CORRUPT", "dataset version does not match its catalog identity",
                      details={"dataset_version_id": dataset_version_id})
        return manifest

    def _contract_ref(self, conn: sqlite3.Connection, contract_id: str) -> TableContractRef:
        row = conn.execute("SELECT definition_hash FROM data_contracts WHERE contract_id = ?",
                           (contract_id,)).fetchone()
        if row is None:
            raise errors.fail("MANIFEST_CORRUPT", "dataset version references an unknown contract",
                      details={"contract_id": contract_id})
        return TableContractRef(contract_id=contract_id, definition_hash=row["definition_hash"])

    def _records(self, conn: sqlite3.Connection, dataset_version_id: str,
                contract_ref: TableContractRef) -> list:
        rows = conn.execute(
            "SELECT f.* FROM data_version_fragments vf JOIN data_fragments f "
            "ON f.fragment_id = vf.fragment_id WHERE vf.dataset_version_id = ? "
            "ORDER BY vf.ordinal", (dataset_version_id,)).fetchall()
        return [self._record(conn, row, contract_ref) for row in rows]

    def _record(self, conn: sqlite3.Connection, row: sqlite3.Row, contract_ref: TableContractRef):
        object_row = conn.execute("SELECT * FROM data_objects WHERE object_id = ?",
                                  (row["object_id"],)).fetchone()
        if object_row is None:
            raise errors.fail("MANIFEST_CORRUPT", "fragment references an unknown object",
                      details={"object_id": row["object_id"]})
        object_ref = ObjectRef(kind=object_row["kind"], object_id=object_row["object_id"],
                               content_hash=object_row["content_hash"],
                               byte_size=object_row["byte_size"])
        bounds = json.loads(row["key_bounds_json"])
        time_bounds = (json.loads(row["time_bounds_json"]) if row["time_bounds_json"] is not None
                      else {"time_min": None, "time_max": None})
        inspection = objects.FragmentInspection(
            object_ref=object_ref, partition_key=row["partition_key"], row_count=row["row_count"],
            byte_hash=row["byte_hash"], logical_content_hash=row["logical_content_hash"],
            primary_key_min=tuple(bounds["primary_key_min"]),
            primary_key_max=tuple(bounds["primary_key_max"]),
            time_min=time_bounds["time_min"], time_max=time_bounds["time_max"])
        record = manifests.fragment_record(
            inspection, contract_ref,
            input_receipt_refs=tuple(json.loads(row["input_receipt_refs_json"])),
            import_request_hash=row["import_request_hash"])
        if record.fragment_id != row["fragment_id"]:
            raise errors.fail("MANIFEST_CORRUPT", "fragment does not match its catalog identity",
                      details={"fragment_id": row["fragment_id"]})
        return record

    # ----------------------------------------------------------------------
    # P2-4: scan — §8.2
    # ----------------------------------------------------------------------

    def scan(self, query: DataQuery, *, table_name: str):
        """A bounded Arrow scan over one table's pinned fragments, per §8.2's
        nine ordered steps. Never a convenience ``read_table()``: this always
        yields batches, never a combined frame."""
        if self._store is None:
            raise RuntimeError("Repository.scan requires an ArtifactStore for object reads")
        validated = self._validated_query(query)
        snap = self.resolve(validated.snapshot_id)
        contract, records = self._table_records(snap, table_name, validated.table_contract_ref)
        query_mod.validate_query(contract, validated)
        yield from self._execute_scan(contract, records, validated)

    def _validated_query(self, query: DataQuery) -> DataQuery:
        """§8.2 step 1: strict structural decode (limits, deadline, bounded-
        ness, sorted/unique ``in`` values, ...) — everything
        ``documents.decode_document`` already knows how to check."""
        try:
            return documents.decode_document(DataQuery, to_document(query))
        except DocumentError as exc:
            raise errors.fail("QUERY_NOT_BOUNDED", f"{exc.code} at {exc.path}: {exc}") from exc

    def _table_records(self, snap: SnapshotRef, table_name: str,
                       table_contract_ref: TableContractRef) -> tuple[TableContract, list]:
        """§8.2 step 2: resolve ``table_name`` under ``snap`` and verify the
        caller's ``table_contract_ref`` pins the same version, then fetch the
        full ordered fragment membership (§8.2 step 4 — never a glob)."""
        if table_name not in snap.table_versions:
            raise errors.fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                      details={"table_name": table_name})
        dvr = snap.table_versions[table_name]
        if table_contract_ref != dvr.table_contract_ref:
            raise errors.fail("CONTRACT_MISMATCH",
                      "table_contract_ref does not match the version pinned under this table name",
                      details={"table_name": table_name})
        with _read_only(self._conn) as conn:
            contract = self._full_contract(conn, dvr.table_contract_ref.contract_id)
            records = self._records(conn, dvr.dataset_version_id, dvr.table_contract_ref)
        return contract, records

    def _full_contract(self, conn: sqlite3.Connection, contract_id: str) -> TableContract:
        row = conn.execute("SELECT definition_json FROM data_contracts WHERE contract_id = ?",
                           (contract_id,)).fetchone()
        if row is None:
            raise errors.fail("MANIFEST_CORRUPT", "dataset version references an unknown contract",
                      details={"contract_id": contract_id})
        return documents.decode_document(TableContract, json.loads(row["definition_json"]))

    def _execute_scan(self, contract: TableContract, records: list[FragmentRecord],
                      query: DataQuery):
        surviving = [r for r in records if query_mod.fragment_may_match(r, contract, query)]
        needed = tuple(dict.fromkeys((*query.columns, *contract.primary_key, *self._hidden_columns(query))))
        batch_cap = min(query.max_batch_rows, contract.maximum_batch_rows)
        streams = [self._fragment_rows(r, contract, needed, query, batch_cap) for r in surviving]
        merged = heapq.merge(*streams, key=lambda item: item[0])
        yield from self._yield_batches(merged, contract, query, batch_cap)
        if query.deadline is not None:
            self._check_deadline(query.deadline)

    def _hidden_columns(self, query: DataQuery) -> tuple:
        """Every ``key_filter``/``time_interval`` column ``query.columns``
        does not itself request (task brief P2-C05, decision 1): a
        predicate the scan must apply has to be READ off every fragment
        even when the caller never asked for that column back — otherwise
        ``query_mod.row_matches``' ``row.get(predicate.column)`` sees a
        column absent from ``row`` entirely (not merely ``None``) and the
        predicate silently excludes every row instead of filtering by it.
        ``_to_batch`` still projects only ``query.columns`` into the
        yielded batch, so a hidden column never reaches the caller."""
        columns = [p.column for p in query.key_filter]
        if query.time_interval is not None:
            columns.append(query.time_interval.column)
        return tuple(columns)

    def _fragment_rows(self, record: FragmentRecord, contract: TableContract, needed: tuple,
                       query: DataQuery, batch_cap: int):
        """§8.2 step 6-7 for one surviving fragment: open, verify, project,
        and filter — lazily, one row at a time."""
        path = objects.verify_object_path(self._store, record.object_ref)
        parquet_file = self._open_parquet(path)
        present, missing = self._match_columns(contract, needed, parquet_file.schema_arrow)
        for batch in self._iter_batches(parquet_file, present, batch_cap):
            for row in self._decode_rows(batch, needed, present, missing):
                if query_mod.row_matches(row, contract, query):
                    yield query_mod.order_key(row, contract), row

    def _open_parquet(self, path) -> pq.ParquetFile:
        try:
            return pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as exc:
            raise errors.fail("OBJECT_CORRUPT", "parquet footer could not be read") from exc

    def _match_columns(self, contract: TableContract, needed: tuple,
                       schema: pa.Schema) -> tuple[list[str], list[str]]:
        file_names = set(schema.names)
        present, missing = [], []
        for name in needed:
            if name in file_names:
                present.append(name)
                continue
            column = next(c for c in contract.columns if c.name == name)
            if not column.nullable:
                raise errors.fail("CONTRACT_MISMATCH", f"required column {name!r} missing from a fragment")
            missing.append(name)
        return present, missing

    def _iter_batches(self, parquet_file: pq.ParquetFile, present: list[str], batch_cap: int):
        try:
            yield from parquet_file.iter_batches(batch_size=batch_cap, columns=present)
        except (OSError, pa.ArrowException) as exc:
            raise errors.fail("OBJECT_CORRUPT", "parquet batch could not be decoded") from exc

    def _decode_rows(self, batch: pa.RecordBatch, needed: tuple, present: list[str],
                     missing: list[str]):
        columns = {name: batch.column(name).to_pylist() for name in present}
        for name in missing:
            columns[name] = [None] * batch.num_rows
        for i in range(batch.num_rows):
            yield {name: columns[name][i] for name in needed}

    def _yield_batches(self, merged, contract: TableContract, query: DataQuery,
                       batch_cap: int):
        cumulative = 0
        previous_key = None
        pending: list[dict] = []
        for key, row in merged:
            # Strictly increasing, unique PK order across the whole stream
            # (§8.2 step 9) — a duplicate key (task 2 review fix) is exactly
            # as much an integrity defect as a regression, so both raise
            # MANIFEST_CORRUPT here, before the duplicate/out-of-order row is
            # ever appended to ``pending`` (so it is never yielded).
            if previous_key is not None and key <= previous_key:
                raise errors.fail("MANIFEST_CORRUPT", "scan result is not in strict primary-key order")
            previous_key = key
            pending.append(row)
            if len(pending) >= batch_cap:
                cumulative = self._check_result_limit(cumulative, len(pending), query.max_result_rows)
                yield self._to_batch(pending, contract, query.columns)
                pending = []
        if pending:
            cumulative = self._check_result_limit(cumulative, len(pending), query.max_result_rows)
            yield self._to_batch(pending, contract, query.columns)

    def _check_result_limit(self, cumulative: int, incoming: int, max_result_rows: int) -> int:
        total = cumulative + incoming
        if total > max_result_rows:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "scan result exceeds max_result_rows",
                      details={"max_result_rows": max_result_rows})
        return total

    def _to_batch(self, rows: list[dict], contract: TableContract, columns: tuple) -> pa.RecordBatch:
        arrays, names = [], []
        for name in columns:
            physical = next(c.physical_type for c in contract.columns if c.name == name)
            arrays.append(pa.array([r[name] for r in rows], type=query_mod.arrow_type_for(physical)))
            names.append(name)
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def _check_deadline(self, deadline: str) -> None:
        if SystemClock().now() > parse_timestamp(deadline):
            raise errors.fail("DEADLINE_EXCEEDED", "deadline exceeded")

    # ----------------------------------------------------------------------
    # P2-4: get_event / get_chain — §8.3
    # ----------------------------------------------------------------------

    def get_event(self, event_ref: EventRef, snapshot_ref: SnapshotRef) -> EarningsEvent:
        return events.get_event(self, event_ref, snapshot_ref)

    def get_chain(self, query: ChainQuery, snapshot_ref: SnapshotRef) -> ChainSnapshot:
        return chains.get_chain(self, query, snapshot_ref)

    # ----------------------------------------------------------------------
    # P2-6: table_contract — the one public contract lookup a query planner
    # needs before it can build a DataQuery (observation_time_column, primary
    # key, per-table row caps). Reuses the same private ``_full_contract``
    # ``resolve``/``scan`` already trust, so a planner never re-derives
    # contract facts by hand.
    # ----------------------------------------------------------------------

    def table_contract(self, snapshot_ref: SnapshotRef, table_name: str) -> TableContract:
        if table_name not in snapshot_ref.table_versions:
            raise errors.fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                      details={"table_name": table_name})
        dvr = snapshot_ref.table_versions[table_name]
        with _read_only(self._conn) as conn:
            return self._full_contract(conn, dvr.table_contract_ref.contract_id)

    def fragment_records(self, snapshot_ref: SnapshotRef, table_name: str) -> tuple[FragmentRecord, ...]:
        """Every ``FragmentRecord`` a table's pinned dataset version carries —
        a query planner's only way to see manifest-derived key/time bounds
        (e.g. a whole-table read's explicit interval, P2-6) without
        duplicating ``resolve``'s own membership walk."""
        if table_name not in snapshot_ref.table_versions:
            raise errors.fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                      details={"table_name": table_name})
        dvr = snapshot_ref.table_versions[table_name]
        with _read_only(self._conn) as conn:
            return tuple(self._records(conn, dvr.dataset_version_id, dvr.table_contract_ref))

    # ----------------------------------------------------------------------
    # P2-4: explain_dependencies — §5.5
    # ----------------------------------------------------------------------

    def explain_dependencies(self, query: DataQuery | ChainQuery, *,
                             table_name: str | None = None) -> DependencyPlan:
        """§5.5: names the exact snapshot, dataset versions, fragments,
        columns, predicates, and estimated/maximum rows a query would touch.
        Recipe/invalidation planning is Phase 3/4 — refused as
        ``UNSUPPORTED_CONTRACT`` here, never a guessed plan (task brief
        decision 5): only ``DataQuery`` is supported in Phase 2; a
        ``ChainQuery`` carries no ``snapshot_id`` to explain against, so it is
        refused the same way.
        """
        if isinstance(query, DataQuery):
            if table_name is None:
                raise errors.fail("UNSUPPORTED_CONTRACT",
                          "explain_dependencies requires table_name for a DataQuery",
                          details={"contract": DATA_QUERY_V1})
            return self._explain_data_query(query, table_name)
        if isinstance(query, ChainQuery):
            raise errors.fail("UNSUPPORTED_CONTRACT",
                      "dependency planning for a chain query is not supported in Phase 2",
                      details={"contract": CHAIN_QUERY_V1})
        raise errors.fail("UNSUPPORTED_CONTRACT", "unrecognized query contract",
                  details={"contract": getattr(query, "schema_version", type(query).__name__)})

    def _explain_data_query(self, query: DataQuery, table_name: str) -> DependencyPlan:
        validated = self._validated_query(query)
        snap = self.resolve(validated.snapshot_id)
        contract, records = self._table_records(snap, table_name, validated.table_contract_ref)
        query_mod.validate_query(contract, validated)
        surviving = [r for r in records if query_mod.fragment_may_match(r, contract, validated)]
        dependencies = tuple(
            DependencyEntry(table_name=table_name, dataset_version_ref=snap.table_versions[table_name],
                            fragment_ref=manifests.fragment_ref(record), columns=validated.columns,
                            predicates=validated.key_filter, estimated_rows=record.row_count,
                            maximum_rows=validated.max_result_rows)
            for record in surviving)
        return DependencyPlan(request_hash=content_hash(to_document(validated)), snapshot_ref=snap,
                              dependencies=dependencies)
