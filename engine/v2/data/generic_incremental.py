"""Durable candidate building for the non-daily EOD table families.

The merge policy lives in :mod:`incremental_tables`; this module supplies the
same immutable-object, manifest, and atomic-head protocol used by
``daily_market`` for any registered ``TableContract``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Callable, Sequence

import pyarrow as pa
from pyarrow import parquet as pq

from engine.v2.contracts import (
    ChangeSet,
    CompletedCoverage,
    ObjectRef,
    RevisionCandidate,
    TableContract,
)
from engine.v2.data import catalog, errors, manifests, objects
from engine.v2.data import incremental_tables as tables
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactStore,
    DocumentError,
    SystemClock,
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    to_document,
)

__all__ = [
    "GenericTableCandidate",
    "build_generic_table_candidate",
    "commit_generic_table_candidate",
    "load_generic_revisions",
]


@dataclass(frozen=True, kw_only=True)
class GenericTableCandidate:
    parent: manifests.ResolvedSnapshot
    table_name: str
    contract: TableContract
    merge: tables.GenericMerge
    coverage: CompletedCoverage
    table_manifest: Any
    snapshot: Any
    contracts: tuple[TableContract, ...]
    objects: tuple[ObjectRef, ...]
    records: tuple[Any, ...]
    changeset: ChangeSet
    changeset_hash: str
    rewritten_partitions: int


def build_generic_table_candidate(
    parent: manifests.ResolvedSnapshot,
    store: ArtifactStore,
    table_name: str,
    incoming: Sequence[tables.GenericRevision],
    *,
    coverage: CompletedCoverage,
    retained: Sequence[tables.GenericRevision] = (),
    parent_snapshot_id: str | None = None,
) -> GenericTableCandidate:
    if coverage.state != "complete":
        raise errors.fail("INPUT_CHANGED", "incomplete coverage cannot build a candidate")
    contract = _contract(parent, table_name)
    prior_manifest = parent.table_manifests.get(table_name)
    if prior_manifest is None:
        raise errors.fail("CONTRACT_MISMATCH", "parent snapshot has no requested table")
    incoming_keys = {item.candidate.logical_key for item in incoming}
    retained = tuple(item for item in retained
                     if item.candidate.logical_key in incoming_keys)
    prior_records = _table_records(parent, prior_manifest)
    affected = _affected_partitions(contract, (*retained, *incoming))
    prior_rows = _load_rows(store, prior_records, contract, partitions=affected)
    merge = tables.merge_table_rows(contract, prior_rows, retained, incoming)
    if merge.changes:
        merge = replace(
            merge,
            changed_partitions=tuple(sorted(
                set(merge.changed_partitions) | set(affected)
            )),
        )
    ref = prior_manifest.dataset_version_ref.table_contract_ref
    new_records, new_objects = _write_partitions(
        store, contract, ref, merge, coverage)
    manifest = _replace_manifest(
        prior_manifest, ref, prior_records, new_records, merge, coverage)
    table_manifests = dict(parent.table_manifests)
    table_manifests[table_name] = manifest
    snapshot = manifests.snapshot_ref(
        table_manifests, calendar_version=parent.snapshot.calendar_version,
        source_priority_version=parent.snapshot.source_priority_version,
        finality_receipt_refs=parent.snapshot.finality_receipt_refs,
        parent_snapshot_id=parent_snapshot_id or parent.snapshot.snapshot_id)
    records = _replace_records(
        parent, prior_records, new_records, merge.changed_partitions,
    )
    objects_by_id = {item.object_id: item for item in parent.objects}
    objects_by_id.update({item.object_id: item for item in new_objects})
    changeset = _changeset(snapshot, prior_manifest, manifest, ref, coverage, merge)
    return GenericTableCandidate(
        parent=parent, table_name=table_name, contract=contract, merge=merge,
        coverage=coverage, table_manifest=manifest, snapshot=snapshot,
        contracts=parent.contracts, objects=tuple(objects_by_id.values()),
        records=records, changeset=changeset,
        changeset_hash=content_hash(to_document(changeset)),
        rewritten_partitions=len(merge.changed_partitions))


def commit_generic_table_candidate(
    conn: Any,
    store: ArtifactStore,
    candidate: GenericTableCandidate,
    *,
    scope: str,
    expected_head_snapshot_id: str | None,
    expected_head_generation: int,
    clock=None,
    request_hash: str | None = None,
    receipt_id: str | None = None,
    attempt_id: str | None = None,
    fence: int = 1,
    fault: Callable[[str], None] | None = None,
):
    clock = clock or SystemClock()
    request_hash = request_hash or content_hash({"changeset": candidate.changeset_hash})
    receipt_id = receipt_id or "receipt_" + request_hash.removeprefix(CONTENT_HASH_PREFIX)[:32]
    attempt_id = attempt_id or "attempt_" + request_hash.removeprefix(CONTENT_HASH_PREFIX)[:32]
    return catalog.commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=candidate.contracts,
        objects=candidate.objects, records=candidate.records,
        manifests=tuple(candidate.parent.table_manifests[name] if name != candidate.table_name
                        else candidate.table_manifest for name in candidate.parent.table_manifests),
        snapshot=candidate.snapshot,
        expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation, receipt_id=receipt_id,
        attempt_id=attempt_id, fence=fence,
        fence_check=lambda c: _head_fence(c, scope, expected_head_snapshot_id,
                                          expected_head_generation),
        clock=clock, fault=fault, store=store,
        record_references=lambda c, rid: _record_references(c, rid, candidate, clock),
        audit_partitions=True)


def load_generic_revisions(conn, table_name: str, contract: TableContract | None = None) -> tuple[tables.GenericRevision, ...]:
    rows = conn.execute(
        "SELECT * FROM data_table_revisions WHERE table_name = ? ORDER BY revision_id",
        (table_name,)).fetchall()
    revisions = []
    for row in rows:
        try:
            candidate = from_document(RevisionCandidate, {
                "schema_version": "revision_candidate.v1.0",
                "revision_id": row["revision_id"],
                "logical_key": row["logical_key"],
                "source": row["source"],
                "source_priority": row["source_priority"],
                "finality": "final" if row["finality_rank"] else "provisional",
                "revision_ordinal": row["revision_number"],
                "received_at": row["created_at"],
                "content_hash": row["row_hash"],
            })
            payload = None if row["row_json"] is None else json.loads(row["row_json"])
            if payload is not None and contract is not None:
                payload = _decode_row(contract, payload)
        except (DocumentError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise errors.fail("MANIFEST_CORRUPT", "generic revision audit row is malformed") from exc
        revisions.append(tables.GenericRevision(
            candidate=candidate, row=payload, deleted=bool(row["deleted"]),
            partition_key=row["partition_key"]))
    return tuple(revisions)


def _decode_row(contract, payload):
    row = dict(payload)
    for column in contract.columns:
        value = row.get(column.name)
        if value is None:
            continue
        if column.physical_type.startswith("timestamp["):
            row[column.name] = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        elif column.physical_type == "date32":
            row[column.name] = date.fromisoformat(str(value)[:10])
    return row


def _contract(parent, table_name):
    try:
        return next(item for item in parent.contracts if item.table_name == table_name)
    except StopIteration:
        raise errors.fail("CONTRACT_MISMATCH", "requested table contract is absent") from None


def _table_records(parent, manifest):
    ids = {item.fragment_id for item in manifest.fragment_refs}
    return tuple(item for item in parent.records if item.fragment_id in ids)


def _load_rows(store, records, contract, *, partitions=None):
    rows = []
    for record in records:
        if partitions is not None and record.partition_key not in partitions:
            continue
        path = objects.verify_object_path(store, record.object_ref)
        rows.extend(pq.read_table(path).to_pylist())
    return tuple(rows)


def _affected_partitions(contract, revisions):
    if not revisions:
        return frozenset()
    partitions = set()
    for revision in revisions:
        if revision.partition_key:
            partitions.add(revision.partition_key)
        if not revision.deleted and revision.row is not None:
            partitions.add(_partition(contract, revision.row))
    return frozenset(partitions)


def _write_partitions(store, contract, contract_ref, merge, coverage):
    records, refs = [], []
    for partition in merge.changed_partitions:
        rows = sorted(
            (row for row in merge.rows if _partition(contract, row) == partition),
            key=lambda row: tuple(_sort_value(row[name]) for name in contract.primary_key))
        if not rows:
            continue
        published = store.publish_bytes(_parquet_bytes(contract, rows),
                                        schema_ref="parquet_fragment.v1.0")
        obj = ObjectRef(kind="parquet_fragment", object_id=published.artifact_id,
                        content_hash=published.content_hash, byte_size=published.byte_size)
        inspection = objects.inspect_fragment(store, obj, contract, contract_ref, partition)
        records.append(manifests.fragment_record(
            inspection, contract_ref,
            input_receipt_refs=coverage.acquisition_receipt_refs,
            import_request_hash=content_hash({
                "table": contract.table_name, "partition": partition,
                "coverage": coverage.coverage_id})))
        refs.append(obj)
    return tuple(records), tuple(refs)


def _replace_manifest(prior, contract_ref, prior_records, new_records, merge, coverage):
    changed = set(merge.changed_partitions)
    kept = [item for item in prior_records if item.partition_key not in changed]
    records = tuple(sorted((*kept, *new_records),
                           key=lambda item: (item.partition_key, item.primary_key_min)))
    hashes = {key: value for key, value in prior.partition_logical_hashes.items()
              if key not in changed}
    return manifests.dataset_manifest(
        contract_ref, records, knowledge_mode=prior.knowledge_mode,
        coverage_receipt_refs=tuple(dict.fromkeys(
            (*prior.coverage_receipt_refs, coverage.coverage_id))),
        availability_evidence_refs=prior.availability_evidence_refs,
        parent_dataset_version_id=prior.dataset_version_ref.dataset_version_id,
        partition_logical_hashes=hashes)


def _replace_records(parent, prior_records, new_records, changed_partitions):
    changed = set(changed_partitions)
    replaced_ids = {item.fragment_id for item in prior_records
                    if item.partition_key in changed}
    kept = [item for item in parent.records if item.fragment_id not in replaced_ids]
    return tuple(sorted((*kept, *new_records),
                        key=lambda item: (item.table_contract_ref.contract_id,
                                          item.partition_key, item.primary_key_min)))


def _changeset(snapshot, prior, result, ref, coverage, merge):
    ident = {"snapshot": snapshot.snapshot_id,
             "table": ref.contract_id,
             "changes": [to_document(item) for item in merge.changes]}
    change_id = "changeset_" + content_hash(ident).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return ChangeSet(
        changeset_id=change_id, table_contract_ref=ref,
        base_dataset_version_ref=prior.dataset_version_ref,
        result_dataset_version_ref=result.dataset_version_ref,
        acquisition_receipt_refs=coverage.acquisition_receipt_refs,
        coverage_receipt_refs=(coverage.coverage_id,), changes=merge.changes,
        changed_partitions=merge.changed_partitions, dependency_impacts=(),
        unknown_dependencies=(), dependency_disposition="exact",
        outcome="changed" if merge.changes else "noop",
        normalized_payloads=len(merge.incoming_revisions) if hasattr(merge, "incoming_revisions") else 0,
        rewritten_partitions=len(merge.changed_partitions))


def _record_references(conn, receipt_id, candidate, clock):
    coverage = candidate.coverage
    coverage_json = canonical_json(to_document(coverage))
    coverage_hash = content_hash(to_document(coverage))
    existing = conn.execute(
        "SELECT coverage_hash, coverage_json FROM data_snapshot_coverage "
        "WHERE snapshot_id = ? AND table_name = ? AND coverage_id = ?",
        (candidate.snapshot.snapshot_id, candidate.table_name, coverage.coverage_id)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO data_snapshot_coverage (snapshot_id, table_name, coverage_id, "
            "import_receipt_id, coverage_hash, coverage_json) VALUES (?, ?, ?, ?, ?, ?)",
            (candidate.snapshot.snapshot_id, candidate.table_name, coverage.coverage_id,
             receipt_id, coverage_hash, coverage_json))
    elif tuple(existing) != (coverage_hash, coverage_json):
        raise errors.fail("IDENTITY_CONFLICT", "coverage identity has conflicting content")
    changeset = candidate.changeset
    payload = canonical_json(to_document(changeset))
    existing = conn.execute(
        "SELECT changeset_hash, changeset_json FROM data_changesets WHERE changeset_id = ?",
        (changeset.changeset_id,)).fetchone()
    values = (changeset.changeset_id, candidate.snapshot.snapshot_id, receipt_id,
              candidate.table_name, changeset.base_dataset_version_ref.dataset_version_id,
              changeset.result_dataset_version_ref.dataset_version_id,
              candidate.changeset_hash, payload, format_timestamp(clock.now()))
    if existing is None:
        conn.execute(
            "INSERT INTO data_changesets (changeset_id, snapshot_id, import_receipt_id, "
            "table_name, old_dataset_version_id, new_dataset_version_id, changeset_hash, "
            "changeset_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
    elif tuple(existing) != (candidate.changeset_hash, payload):
        raise errors.fail("IDENTITY_CONFLICT", "changeset identity has conflicting content")
    for revision in candidate.merge.incoming_revisions:
        row_json = (None if revision.row is None else
                    canonical_json(to_document(dict(revision.row))))
        values = (
            revision.candidate.revision_id, receipt_id, candidate.table_name,
            revision.candidate.logical_key, revision.partition_key,
            revision.candidate.source, revision.candidate.source_priority,
            int(revision.candidate.finality == "final"),
            revision.candidate.revision_ordinal, int(revision.deleted), row_json,
            revision.candidate.content_hash, format_timestamp(clock.now()))
        prior = conn.execute(
            "SELECT import_receipt_id, table_name, logical_key, partition_key, source, "
            "source_priority, finality_rank, revision_number, deleted, row_json, row_hash "
            "FROM data_table_revisions WHERE revision_id = ? AND table_name = ?",
            (revision.candidate.revision_id, candidate.table_name)).fetchone()
        if prior is None:
            conn.execute(
                "INSERT INTO data_table_revisions (revision_id, import_receipt_id, table_name, "
                "logical_key, partition_key, source, source_priority, finality_rank, "
                "revision_number, deleted, row_json, row_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        elif tuple(prior[1:]) != values[2:-1]:
            raise errors.fail("IDENTITY_CONFLICT", "generic revision identity has conflicting content")


def _head_fence(conn, scope, expected_snapshot, expected_generation):
    row = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?", (scope,)
    ).fetchone()
    actual = None if row is None else (row["snapshot_id"], row["generation"])
    if actual != (expected_snapshot, expected_generation):
        raise errors.fail("SNAPSHOT_CONFLICT", "generic refresh lost its parent head")


def _partition(contract, row):
    if not contract.partition_columns:
        return "__whole__"
    return "/".join(str(row[name]) for name in contract.partition_columns)


def _sort_value(value):
    if value is None:
        return (0, None)
    return (1, value)


def _parquet_bytes(contract, rows):
    types = {"string": pa.string(), "float64": pa.float64(), "int64": pa.int64(),
             "bool": pa.bool_(), "timestamp[ns]": pa.timestamp("ns"),
             "timestamp[us]": pa.timestamp("us"), "date32": pa.date32()}
    arrays = {}
    for column in contract.columns:
        arrow_type = types.get(column.physical_type)
        if arrow_type is None:
            raise errors.fail("UNSUPPORTED_CONTRACT", "table physical type is unsupported")
        arrays[column.name] = pa.array([row.get(column.name) for row in rows], type=arrow_type)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(arrays), sink)
    return sink.getvalue().to_pybytes()
