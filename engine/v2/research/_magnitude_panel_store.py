"""Immutable qualified panel artifact transport; private orchestration helpers."""
import math

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import DataQuery, KeyPredicate, ObjectRef, TableContractRef
from engine.v2.data import catalog, manifests, objects, query
from engine.v2.data.errors import fail
from engine.v2.foundation import SystemClock, content_hash
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research._snapshot import artifact_store, catalog_connection

_FRAGMENT_ROWS = 3  # one event's horizons per stored fragment


def _slot(context):
    return "qualified-size-panel-" + context["context_id"].removeprefix("sha256:")


def existing(repository, context):
    slot = _slot(context)
    row = catalog_connection(repository).execute(
        """SELECT CASE WHEN scope=? AND source_manifest_hash=? AND attempt_id=?
        AND fence=1 AND status='committed' AND length(result_snapshot_id)=37
        AND substr(result_snapshot_id,1,5)='snap_'
        AND substr(result_snapshot_id,6) NOT GLOB '*[^0-9a-f]*'
        THEN result_snapshot_id END AS snapshot_id FROM data_import_receipts WHERE receipt_id=?""",
        (slot, context["context_id"], slot, slot)).fetchone()
    if row is None:
        return None
    if row["snapshot_id"] is None:
        raise fail("IDENTITY_CONFLICT", "qualified panel receipt differs")
    return repository.resolve(row["snapshot_id"])


def _validate(rows, context):
    expected = {(t["event_id"], c["horizon_sessions"]): (t, c)
                for t in context["target_events"] for c in t["forecasts"]}
    ordered = sorted(expected)

    def validated():
        position = 0
        for row in rows:
            key = row["event_id"], row["horizon_sessions"]
            if position >= len(ordered) or key != ordered[position] or set(row) != {c.name for c in table.CONTRACT.columns}:
                raise fail("CONTRACT_MISMATCH", "qualified panel cohort differs")
            target, cutoff = expected[key]
            _validate_row(row, context, target, cutoff)
            position += 1
            yield row
        if position != len(ordered):
            raise fail("CONTRACT_MISMATCH", "qualified panel cohort incomplete")

    return validated()


def _validate_row(row, context, target, cutoff):
    pins = {"schema_version": table.CONTRACT.contract_id, "context_id": context["context_id"],
            "history_context_id": context["history_context"]["context_id"],
            "event_date": target["event_date"], "cutoff_id": cutoff["cutoff_id"],
            "source_session_ceiling": cutoff["source_session_ceiling"]}
    if (any(row[k] != v for k, v in pins.items())
            or row["row_id"] != content_hash({k: v for k, v in row.items() if k != "row_id"})
            or row["current_quote_status"] not in ("present", "zero", "absent", "unavailable")
            or row["has_implied_quote"] != int(row["current_quote_status"] == "present")
            or (row["implied_move"] is not None) != bool(row["has_implied_quote"])):
        raise fail("CONTRACT_MISMATCH", "qualified panel identity or quote status differs")
    for name in table.FEATURES:
        value = row[name]
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
            raise fail("CONTRACT_MISMATCH", "qualified panel feature is not finite")


def publish(repository, source, context, rows):
    limits = context["request"]["limits"]
    # Explicit row bound is checked by history admission before construction.
    rows = iter(_validate(rows, context))
    store, records, total = artifact_store(repository), [], 0
    ref = TableContractRef(contract_id=table.CONTRACT.contract_id, definition_hash=table.CONTRACT.definition_hash)
    schema = pa.schema([pa.field(c.name, query.arrow_type_for(c.physical_type), nullable=c.nullable)
                        for c in table.CONTRACT.columns])
    def publish_chunk(chunk):
        nonlocal total
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(chunk, schema=schema), sink)
        payload = sink.getvalue().to_pybytes()
        total += len(payload)
        if len(payload) > limits["max_object_bytes"] or total > limits["max_total_bytes"]:
            raise fail("RESULT_LIMIT_EXCEEDED", "panel output exceeds physical byte limits")
        cap = min(limits["max_object_bytes"], 2**31 - 1)
        with pq.ParquetFile(pa.BufferReader(payload), thrift_string_size_limit=cap,
                            thrift_container_size_limit=cap) as parquet:
            expanded = sum(parquet.metadata.row_group(i).total_byte_size
                           for i in range(parquet.metadata.num_row_groups))
            if expanded > limits["max_object_bytes"]:
                raise fail("RESULT_LIMIT_EXCEEDED", "panel output exceeds uncompressed object byte limit")
        published = store.publish_bytes(payload, schema_ref="parquet_fragment.v1")
        obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                        content_hash=published.content_hash, byte_size=published.byte_size)
        inspected = objects.inspect_fragment(store, obj, table.CONTRACT, ref, context["context_id"])
        records.append(manifests.fragment_record(inspected, ref, input_receipt_refs=(),
                                                import_request_hash=context["context_id"]))

    chunk = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == _FRAGMENT_ROWS:
            publish_chunk(chunk)
            chunk = []
    if chunk:
        publish_chunk(chunk)
    logical = objects.partition_logical_hash(store, [r.object_ref for r in records], table.CONTRACT,
                                             ref, context["context_id"])
    manifest = manifests.dataset_manifest(ref, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        partition_logical_hashes={context["context_id"]: logical})
    snapshot = manifests.snapshot_ref({table.TABLE: manifest}, calendar_version=source.calendar_version,
        source_priority_version=source.source_priority_version, finality_receipt_refs=())
    slot = _slot(context)
    committed = catalog.commit_snapshot(catalog_connection(repository), scope=slot,
        request_hash=context["context_id"], contracts=(table.CONTRACT,),
        objects=tuple(r.object_ref for r in records), records=records, manifests=(manifest,), snapshot=snapshot,
        expected_head_snapshot_id=None, expected_head_generation=0, receipt_id=slot, attempt_id=slot, fence=1,
        fence_check=lambda conn: None, clock=SystemClock(), store=store)
    return committed.snapshot_ref


def _bounded_query(repository, snapshot, context):
    limits = context["request"]["limits"]
    count = len(context["target_events"]) * len(context["request"]["horizons"])
    records = repository.fragment_records(snapshot, table.TABLE)
    if (sum(r.row_count for r in records) != count
            or any(r.object_ref.byte_size > limits["max_object_bytes"] for r in records)
            or sum(r.object_ref.byte_size for r in records) > limits["max_total_bytes"]):
        raise fail("RESULT_LIMIT_EXCEEDED", "stored panel exceeds explicit bounds")
    return DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions[table.TABLE].table_contract_ref,
        columns=tuple(c.name for c in table.CONTRACT.columns), order_by=table.CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="context_id", operator="eq", values=(context["context_id"],)),),
        max_batch_rows=min(count, max((r.row_count for r in records), default=0),
                           table.CONTRACT.maximum_batch_rows), max_result_rows=count)


def read(repository, snapshot, context, columns, lineage=True):
    if (existing(repository, context) != snapshot or set(snapshot.table_versions) != {table.TABLE}
            or repository.table_contract(snapshot, table.TABLE) != table.CONTRACT):
        raise fail("HOLDOUT_ACCESS_DENIED", "input is not the exact qualified panel artifact")
    request = _bounded_query(repository, snapshot, context)
    # Stream: lineage cells are multi-MB, so validate each row and keep only its projection.
    batches = (r for b in repository.scan(request, table_name=table.TABLE,
        max_object_bytes=context["request"]["limits"]["max_object_bytes"]) for r in b.to_pylist())
    keep = (*(m for m in table.METADATA if lineage or m != "lineage"), *columns)
    return [{k: row[k] for k in keep} for row in _validate(batches, context)]
