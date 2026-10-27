"""Pure native identity reconstruction from an already bounded catalog transfer."""
from __future__ import annotations

from engine.v2.contracts.data import ObjectRef, SnapshotRef, TableContract, TableContractRef
from engine.v2.data import documents, errors, manifests, objects
from engine.v2.data._bounded_catalog_transfer import CatalogPayload
from engine.v2.foundation import DocumentError, to_document


def _require(condition):
    if not condition:
        raise errors.fail("MANIFEST_CORRUPT", "catalog identity does not match")


def _decoded(value):
    return documents.decode_document(type(value), to_document(value))


def _contract(row, snapshot_id):
    contract = documents.decode_document(TableContract, to_document(row["c.definition_json"]))
    _require(row["t.snapshot_id"] == snapshot_id
             and row["t.dataset_version_id"] == row["d.dataset_version_id"]
             and row["d.contract_id"] == row["c.contract_id"] == contract.contract_id
             and row["t.table_name"] == row["c.table_name"] == contract.table_name
             and row["c.schema_version"] == contract.schema_version
             and row["c.definition_hash"] == contract.definition_hash
             == manifests.table_contract_hash(contract))
    return contract, TableContractRef(contract_id=contract.contract_id,
                                      definition_hash=contract.definition_hash)


def _fragment(row, contract_ref):
    _require(row["m.fragment_id"] == row["f.fragment_id"]
             and row["f.object_id"] == row["o.object_id"]
             and row["f.contract_id"] == contract_ref.contract_id)
    obj = ObjectRef(kind=row["o.kind"], object_id=row["o.object_id"],
                    content_hash=row["o.content_hash"], byte_size=row["o.byte_size"])
    keys, times = row["f.key_bounds_json"], row["f.time_bounds_json"]
    inspection = objects.FragmentInspection(
        object_ref=obj, partition_key=row["f.partition_key"], row_count=row["f.row_count"],
        byte_hash=row["f.byte_hash"], logical_content_hash=row["f.logical_content_hash"],
        primary_key_min=keys.primary_key_min, primary_key_max=keys.primary_key_max,
        time_min=None if times is None else times.time_min,
        time_max=None if times is None else times.time_max)
    record = _decoded(manifests.fragment_record(
        inspection, contract_ref, input_receipt_refs=row["f.input_receipt_refs_json"],
        import_request_hash=row["f.import_request_hash"]))
    _require(record.fragment_id == row["f.fragment_id"])
    return record


def _table(row, records, contract_ref):
    evidence = row["d.evidence_json"]
    manifest = _decoded(manifests.dataset_manifest(
        contract_ref, records, knowledge_mode=row["d.knowledge_mode"],
        coverage_receipt_refs=evidence.coverage_receipt_refs,
        availability_evidence_refs=evidence.availability_evidence_refs,
        parent_dataset_version_id=row["d.parent_dataset_version_id"],
        partition_logical_hashes=row["d.partition_logical_hashes_json"]))
    ref = manifest.dataset_version_ref
    _require(ref.dataset_version_id == row["d.dataset_version_id"]
             and ref.manifest_hash == row["d.manifest_hash"]
             and manifest.logical_content_hash == row["d.logical_content_hash"]
             and manifest.row_count == row["d.row_count"])
    return manifest


def reconstruct_catalog(payload: CatalogPayload, expected: SnapshotRef) -> manifests.ResolvedSnapshot:
    """Rebuild every table, then compare the complete expected native reference.

    Private B1b-to-B2 boundary, not a provenance token or a public session. The
    caller supplies an unmodified transfer from its single admitted read view.
    No connection, object store, logging, retry or partial result is involved.
    Native DataErrors propagate with their complete diagnostic envelopes.
    """
    try:
        if type(expected) is not SnapshotRef:
            raise errors.fail("SNAPSHOT_UNRESOLVED", "invalid expected snapshot")
        decoded = _decoded(expected)
        if decoded != expected:
            raise ValueError
        expected = decoded
    except (DocumentError, ValueError, TypeError, RecursionError):
        raise errors.fail("SNAPSHOT_UNRESOLVED", "invalid expected snapshot") from None
    try:
        return _reconstruct(payload, expected)
    except (DocumentError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise errors.fail("MANIFEST_CORRUPT", "catalog identity malformed") from None


def _reconstruct(payload, expected):
    snapshot_row = payload.snapshot
    _require(len(payload.tables) == payload.probe.tables
             and len(payload.fragments) == payload.probe.catalog_fragments)
    contracts, object_refs, all_records, table_manifests = {}, {}, [], {}
    position = 0
    previous_name = None
    for row in payload.tables:
        name = row["t.table_name"]
        _require(previous_name is None or previous_name < name)
        previous_name = name
        contract, contract_ref = _contract(row, snapshot_row["s.snapshot_id"])
        _require(contract.contract_id not in contracts or contracts[contract.contract_id] == contract)
        contracts[contract.contract_id] = contract
        records = []
        while position < len(payload.fragments):
            fragment_row = payload.fragments[position]
            if fragment_row["m.dataset_version_id"] != row["d.dataset_version_id"]:
                break
            _require(fragment_row["m.ordinal"] == len(records))
            record = _fragment(fragment_row, contract_ref)
            obj = record.object_ref
            _require(obj.object_id not in object_refs or object_refs[obj.object_id] == obj)
            object_refs[obj.object_id] = obj
            records.append(record)
            position += 1
        table_manifests[name] = _table(row, records, contract_ref)
        all_records.extend(records)
    _require(position == len(payload.fragments))
    snapshot = _decoded(manifests.snapshot_ref(
        table_manifests, calendar_version=snapshot_row["s.calendar_version"],
        source_priority_version=snapshot_row["s.source_priority_version"],
        finality_receipt_refs=snapshot_row["s.finality_receipt_refs_json"],
        parent_snapshot_id=snapshot_row["s.parent_snapshot_id"]))
    _require(snapshot.snapshot_id == snapshot_row["s.snapshot_id"]
             and snapshot.manifest_hash == snapshot_row["s.manifest_hash"]
             and snapshot.knowledge_mode_by_table == snapshot_row["s.knowledge_mode_by_table_json"])
    if snapshot != expected:
        raise errors.fail("SNAPSHOT_UNRESOLVED", "expected snapshot differs")
    return manifests.ResolvedSnapshot(
        contracts=tuple(contracts.values()), objects=tuple(object_refs.values()),
        records=tuple(all_records), table_manifests=table_manifests, snapshot=snapshot)
