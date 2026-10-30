"""Bounded publication of supplied metadata and event-isolated outcome rows."""
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import ObjectRef, TableContract, TableContractRef
from engine.v2.data import catalog, documents, errors, manifests, objects, query
from engine.v2.foundation import (
    DocumentError,
    SystemClock,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.research._snapshot import artifact_store, catalog_connection
from engine.v2.research.progress import Progress, stage
from engine.v2.research.snapshot_inputs import snapshot_reference


@dataclass(frozen=True)
class _Provenance:
    value: Any


def _validate_native_contract(contract):
    try:
        documents.decode_document(TableContract, to_document(contract))
    except DocumentError as exc:
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires structurally valid contracts") from exc


def _contracts(events, target):
    for contract in (events, target):
        if (type(contract) is not TableContract or len(contract.partition_columns) != 1
                or type(contract.maximum_batch_rows) is not int or contract.maximum_batch_rows <= 0
                or contract.definition_hash != manifests.table_contract_hash(contract)):
            raise errors.fail("CONTRACT_MISMATCH", "event publication requires exact table contracts")
        _validate_native_contract(contract)
    _key_types(events, target)
    if (events.table_name != "earnings_events" or events.primary_key != ("event_id",)
            or not {"event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"}
                <= {c.name for c in events.columns}
            or target.table_name == events.table_name or target.primary_key != ("ticker", "event_date")
            or target.partition_columns != ("ticker",) or target.observation_time_column != "event_date"):
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires metadata and ticker/date targets")


def _key_types(events, target):
    for contract, date_type in ((events, "timestamp[ns]"), (target, "string")):
        types = {c.name: c.physical_type for c in contract.columns}
        if types.get("event_date") != date_type or types.get("ticker") != "string":
            raise errors.fail("CONTRACT_MISMATCH", "event publication requires canonical key types")


def _key(row, contract):
    key = tuple(row[name] for name in contract.primary_key)
    if any(type(value) is not str or not value for value in key):
        raise errors.fail("CONTRACT_MISMATCH", "event publication keys must be nonempty strings")
    return key


def _rows(values, contract, limits, budget):
    names = {c.name for c in contract.columns}
    rows, seen = [], set()
    for row in values:
        if len(rows) >= limits["max_rows"]:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "event publication exceeds row cap")
        if type(row) is not dict or set(row) != names:
            raise errors.fail("CONTRACT_MISMATCH", "event publication requires all declared columns exactly")
        if any(row[c.name] is None for c in contract.columns if not c.nullable):
            raise errors.fail("CONTRACT_MISMATCH", "event publication requires nonnull declared values")
        size = len(canonical_json(row).encode("utf-8"))
        budget[0] += size
        if size > limits["max_object_bytes"] or budget[0] > limits["max_total_bytes"]:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "event publication input exceeds byte cap")
        key = _key(row, contract)
        if key in seen:
            raise errors.fail("CONTRACT_MISMATCH", "event publication keys must be unique")
        seen.add(key)
        rows.append(dict(row))
    if not rows:
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires nonempty tables")
    return sorted(rows, key=lambda row: (str(row[contract.partition_columns[0]]),
                                        tuple(row[name] for name in contract.primary_key)))


def _event_keys(rows):
    try:
        if any(r["event_date"].isoformat() != r["event_date"].date().isoformat() + "T00:00:00" for r in rows):
            raise ValueError
        return {(r["ticker"], r["event_date"].date().isoformat()) for r in rows}
    except (AttributeError, TypeError, ValueError):
        raise errors.fail("CONTRACT_MISMATCH", "event metadata requires timestamp dates") from None


def publish_event_targets(repository, *, source_snapshot, provenance, events_contract, events,
                          target_contract, targets, max_rows, max_object_bytes, max_total_bytes):
    """Publish a standalone two-table reconstruction, not authenticated source data.

    Caller owns source/holdout admission and output workspace isolation. Source
    identity and provenance are assertions only. No source repository is read.
    Caps bound each input table and both serialized rows and output Parquet bytes.
    Failed attempts can leave unreferenced immutable objects, never a partial snapshot.
    """
    snapshot_reference(source_snapshot)
    _contracts(events_contract, target_contract)
    limits = dict(max_rows=max_rows, max_object_bytes=max_object_bytes, max_total_bytes=max_total_bytes)
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires explicit positive caps")
    if type(provenance) is not dict or not provenance:
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires caller provenance")
    try:
        provenance = from_document(_Provenance, {"value": provenance}).value
    except (DocumentError, RecursionError):
        raise errors.fail("CONTRACT_MISMATCH", "event publication requires JSON provenance with string keys") from None
    with stage("event target preparation", report_refusals=True):
        budget = [0]
        event_rows = _rows(events, events_contract, limits, budget)
        target_rows = _rows(targets, target_contract, limits, budget)
        event_keys = _event_keys(event_rows)
        if not {(r["ticker"], r["event_date"]) for r in target_rows} <= event_keys:
            raise errors.fail("CONTRACT_MISMATCH", "targets must belong to supplied metadata")
        identity = content_hash(dict(schema_version="event_target_publication.v1",
            source_snapshot=to_document(source_snapshot), provenance=provenance, limits=limits,
            contracts=[to_document(c) for c in (events_contract, target_contract)],
            events=event_rows, targets=target_rows))
    return _publish(repository, source_snapshot, events_contract, event_rows, target_contract, target_rows,
                    identity, max_object_bytes, max_total_bytes)


def _publish(repository, source_snapshot, events_contract, event_rows, target_contract, target_rows,
             identity, max_object_bytes, max_total_bytes):
    store, records, tables, total = artifact_store(repository), [], {}, 0
    progress = Progress("event target rows", len(event_rows) + len(target_rows))
    with stage("event target publication", report_refusals=True):
        for contract, rows in ((events_contract, event_rows), (target_contract, target_rows)):
            ref = TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)
            schema = pa.schema([pa.field(c.name, query.arrow_type_for(c.physical_type), nullable=c.nullable)
                                for c in contract.columns])
            group, partitions = [], {}
            size = 1 if contract == target_contract else min(256, contract.maximum_batch_rows)
            for chunk in _chunks(rows, contract.partition_columns[0], size):
                sink = pa.BufferOutputStream()
                pq.write_table(pa.Table.from_pylist(chunk, schema=schema), sink)
                payload = sink.getvalue().to_pybytes()
                total += len(payload)
                if len(payload) > max_object_bytes or total > max_total_bytes:
                    raise errors.fail("RESULT_LIMIT_EXCEEDED", "event publication output exceeds byte cap")
                published = store.publish_bytes(payload, schema_ref="parquet_fragment.v1")
                obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                                content_hash=published.content_hash, byte_size=published.byte_size)
                partition = str(chunk[0][contract.partition_columns[0]])
                inspection = objects.inspect_fragment(store, obj, contract, ref, partition)
                group.append(manifests.fragment_record(inspection, ref, input_receipt_refs=(),
                                                       import_request_hash=identity))
                partitions.setdefault(partition, []).append(obj)
                progress.tick(len(chunk))
            hashes = {key: objects.partition_logical_hash(store, values, contract, ref, key)
                      for key, values in partitions.items()}
            tables[contract.table_name] = manifests.dataset_manifest(ref, group, knowledge_mode="reconstructed",
                coverage_receipt_refs=(), availability_evidence_refs=(), partition_logical_hashes=hashes)
            records.extend(group)
        snapshot = manifests.snapshot_ref(tables, calendar_version=source_snapshot.calendar_version,
            source_priority_version=source_snapshot.source_priority_version, finality_receipt_refs=())
        name = "event-target-" + identity.removeprefix("sha256:")
        result = catalog.commit_snapshot(catalog_connection(repository), scope=name, request_hash=identity,
            contracts=(events_contract, target_contract), objects=tuple(r.object_ref for r in records),
            records=records, manifests=tuple(tables.values()), snapshot=snapshot,
            expected_head_snapshot_id=None, expected_head_generation=0, receipt_id=name, attempt_id=name,
            fence=1, fence_check=lambda conn: None, clock=SystemClock(), store=store)
    return repository.resolve(result.snapshot_ref.snapshot_id)


def _chunks(rows, partition, size):
    chunk = []
    for row in rows:
        if chunk and (len(chunk) >= size or row[partition] != chunk[0][partition]):
            yield chunk
            chunk = []
        chunk.append(row)
    if chunk:
        yield chunk
