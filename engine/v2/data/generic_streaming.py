"""Streaming whole-table writer for #876 slice 2a.

An additive companion to :mod:`generic_incremental`: a producer whose output
must not be held in memory appends rows in batches and this writer stages one
parquet fragment in the attempt's staging directory as fixed-size row groups.
``finish()`` returns a :class:`GenericTableCandidate` that the unchanged
``commit_generic_table_candidate`` commits; nothing is visible to the catalog
until that one commit.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
from pyarrow import parquet as pq

from engine.v2.contracts import ChangeSet, CompletedCoverage, ObjectRef, TableContract
from engine.v2.data import errors, manifests, objects
from engine.v2.data import generic_incremental as incremental
from engine.v2.data import incremental_tables as tables
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactStore,
    content_hash,
    to_document,
)

__all__ = ["ROW_GROUP_ROWS", "StreamingTableWriter"]

#: Rows per parquet row group; the bytes therefore depend only on the rows.
ROW_GROUP_ROWS = 50_000

#: Same arrow type map as ``generic_incremental._parquet_bytes``.
_ARROW_TYPES = {
    "string": pa.string(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "timestamp[ns]": pa.timestamp("ns"),
    "timestamp[us]": pa.timestamp("us"),
    "date32": pa.date32(),
}


def _arrow_schema(contract: TableContract) -> pa.Schema:
    fields = []
    for column in contract.columns:
        arrow_type = _ARROW_TYPES.get(column.physical_type)
        if arrow_type is None:
            raise errors.fail("UNSUPPORTED_CONTRACT", "table physical type is unsupported")
        fields.append(pa.field(column.name, arrow_type))
    return pa.schema(fields)


class StreamingTableWriter:
    """Stage one whole-table parquet fragment from a sequence of row batches."""

    def __init__(self, store: ArtifactStore, parent: manifests.ResolvedSnapshot,
                 table_name: str, attempt_id: str, coverage: CompletedCoverage) -> None:
        if coverage.state != "complete":
            raise errors.fail("INPUT_CHANGED", "incomplete coverage cannot build a candidate")
        contract = incremental._contract(parent, table_name)
        prior_manifest = parent.table_manifests.get(table_name)
        if prior_manifest is None:
            raise errors.fail("CONTRACT_MISMATCH", "parent snapshot has no requested table")
        if contract.partition_columns != ():
            raise errors.fail("UNSUPPORTED_CONTRACT",
                              "the streaming writer supports whole-table contracts only")
        self._store = store
        self._parent = parent
        self._table_name = table_name
        self._attempt_id = attempt_id
        self._coverage = coverage
        self._contract = contract
        self._prior_manifest = prior_manifest
        self._schema = _arrow_schema(contract)
        self._buffer: list[Mapping[str, Any]] = []
        self._previous_key: tuple | None = None
        self._row_count = 0
        self._state = "open"
        staging = store.staging_dir(attempt_id)
        self._rel = "generic_stream_" + uuid.uuid4().hex + ".parquet"
        self._path = staging / self._rel
        self._writer = pq.ParquetWriter(str(self._path), self._schema)

    # -- lifecycle ----------------------------------------------------------

    def append_rows(self, rows: Sequence[Mapping]) -> None:
        self._ensure_open()
        try:
            for row in rows:
                self._accept(row)
        except BaseException:
            self._fail()
            raise

    def finish(self) -> incremental.GenericTableCandidate:
        self._ensure_open()
        try:
            if self._row_count == 0:
                raise errors.fail("CONTRACT_MISMATCH",
                                  "a streaming table writer cannot finish with no rows")
            if self._buffer:
                self._write_row_group()
            self._writer.close()
            candidate = self._build_candidate()
            self._state = "finished"
            return candidate
        except BaseException:
            self._fail()
            raise

    def abort(self) -> None:
        if self._state != "open":
            return
        self._close_quietly()
        self._remove_staging()
        self._state = "aborted"

    def __enter__(self) -> "StreamingTableWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
        return False

    # -- internals ----------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise errors.fail("INPUT_CHANGED", "the streaming writer is no longer open")

    def _accept(self, row: Mapping) -> None:
        tables._validate_columns(self._contract, row)
        key = tuple(incremental._sort_value(row[name])
                    for name in self._contract.primary_key)
        if self._previous_key is not None and key <= self._previous_key:
            raise errors.fail("CONTRACT_MISMATCH",
                              "streamed rows are not in strictly increasing primary-key order")
        self._previous_key = key
        self._buffer.append(row)
        self._row_count += 1
        if len(self._buffer) == ROW_GROUP_ROWS:
            self._write_row_group()

    def _write_row_group(self) -> None:
        table = pa.Table.from_arrays(
            [pa.array([row.get(column.name) for row in self._buffer],
                      type=_ARROW_TYPES[column.physical_type])
             for column in self._contract.columns],
            schema=self._schema)
        self._writer.write_table(table, row_group_size=len(self._buffer))
        self._buffer.clear()

    def _build_candidate(self) -> incremental.GenericTableCandidate:
        parent = self._parent
        contract = self._contract
        coverage = self._coverage
        prior_manifest = self._prior_manifest
        prior_records = incremental._table_records(parent, prior_manifest)
        ref = prior_manifest.dataset_version_ref.table_contract_ref
        published = self._store.publish_candidate(
            self._attempt_id, self._rel, schema_ref="parquet_fragment.v1.0")
        obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                        content_hash=published.content_hash, byte_size=published.byte_size)
        inspection = objects.inspect_fragment(self._store, obj, contract, ref, "__whole__")
        record = manifests.fragment_record(
            inspection, ref,
            input_receipt_refs=coverage.acquisition_receipt_refs,
            import_request_hash=content_hash({
                "table": contract.table_name, "partition": "__whole__",
                "coverage": coverage.coverage_id}))
        self._remove_staging()
        new_records = (record,)
        merge = tables.GenericMerge(
            rows=(), retained_revisions=(), incoming_revisions=(), winners=(),
            changes=(), changed_partitions=("__whole__",))
        manifest = incremental._replace_manifest(
            prior_manifest, ref, prior_records, new_records, merge, coverage)
        table_manifests = dict(parent.table_manifests)
        table_manifests[self._table_name] = manifest
        snapshot = manifests.snapshot_ref(
            table_manifests, calendar_version=parent.snapshot.calendar_version,
            source_priority_version=parent.snapshot.source_priority_version,
            finality_receipt_refs=parent.snapshot.finality_receipt_refs,
            parent_snapshot_id=parent.snapshot.snapshot_id)
        records = incremental._replace_records(
            parent, prior_records, new_records, merge.changed_partitions)
        objects_by_id = {item.object_id: item for item in parent.objects}
        objects_by_id[obj.object_id] = obj
        changeset = _whole_changeset(
            snapshot, prior_manifest, manifest, ref, coverage, self._row_count)
        return incremental.GenericTableCandidate(
            parent=parent, table_name=self._table_name, contract=contract, merge=merge,
            coverage=coverage, table_manifest=manifest, snapshot=snapshot,
            contracts=parent.contracts, objects=tuple(objects_by_id.values()),
            records=records, changeset=changeset,
            changeset_hash=content_hash(to_document(changeset)), rewritten_partitions=1)

    def _fail(self) -> None:
        if self._state != "open":
            return
        self._state = "failed"
        self._close_quietly()
        self._remove_staging()

    def _close_quietly(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is not None:
            self._writer = None
            try:
                writer.close()
            except Exception:
                pass

    def _remove_staging(self) -> None:
        path = getattr(self, "_path", None)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _whole_changeset(snapshot, prior, result, ref, coverage, row_count) -> ChangeSet:
    ident = {"snapshot": snapshot.snapshot_id, "table": ref.contract_id, "changes": []}
    change_id = "changeset_" + content_hash(ident).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return ChangeSet(
        changeset_id=change_id, table_contract_ref=ref,
        base_dataset_version_ref=prior.dataset_version_ref,
        result_dataset_version_ref=result.dataset_version_ref,
        acquisition_receipt_refs=coverage.acquisition_receipt_refs,
        coverage_receipt_refs=(coverage.coverage_id,), changes=(),
        changed_partitions=("__whole__",), dependency_impacts=(),
        unknown_dependencies=(), dependency_disposition="exact", outcome="changed",
        normalized_payloads=row_count, rewritten_partitions=1)
