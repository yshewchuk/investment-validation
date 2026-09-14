"""§7.3/§10 coordinator effects: commit an import, validate a candidate,
compare, promote, and roll back — task brief P2-7/Task7b.

Both coordinator effects below follow the ``publication_effect`` shape
(``engine/v2/ops/effects_graph.py``'s own docstring): the real work — a
``catalog.commit_snapshot``-family transaction, or a filesystem confinement
check — runs here, entirely *before* ``supervisor.Service._finish`` calls
``lifecycle.commit_attempt``, because ``catalog.commit_snapshot``/
``record_failed_import``/``move_head`` each open their own short
``BEGIN IMMEDIATE`` transaction and nesting is refused (``catalog.py``'s own
rule). The closure this module returns to the caller is therefore always
``None`` — a no-op — and every artifact this module already published and
registered is durable by the time ``_coordinator_effect`` returns.

§7.3: "No file copy, Arrow scan, hash calculation, network call, or legacy
scorer runs inside the transaction." True here exactly as it is in
``catalog.commit_snapshot`` itself: every ``objects.publish_legacy_file``/
``objects.partition_logical_hash`` call below runs before
``snapshots.commit_snapshot_for_attempt`` is ever invoked, which itself
re-verifies everything again before opening its own transaction.

Reference inputs (guide §14): the coordinator publishes every reference file
the import manifest pinned (``reference_inputs.publish_reference_inputs``),
outside the transaction like every other object, and ``commit_snapshot``
inserts one ``data_import_reference_inputs`` row per file in the same
transaction as the receipt. They are recorded per receipt, never folded into
the snapshot: a re-import of identical data with a new model file reuses the
snapshot and records the new refs.

§7.3: "The Phase 1 attempt receipt always records a failed attempt. Once the
coordinator regains control it also publishes a failed SnapshotImportReceipt
artifact and inserts it in a separate short transaction." —
:func:`snapshot_import_effect` catches any failure from its own commit
attempt, calls ``catalog.record_failed_import`` (its own transaction) with
the *data-layer* ``Problem`` it caught, then re-raises an ``OpsError`` whose
code ``supervisor._finish``'s generic handler can register on the attempt
itself; both together satisfy the guide sentence.

§10: a candidate is validated (write confinement) and compared before any
head change; promotion is an explicit, separate coordinator action
(:func:`promote`), never a side effect of a worker or a ``snapshot_import``
succeeding on its own; rollback (:func:`rollback`) is a bare compare-and-swap
to a prior committed snapshot, never a delete or a manifest rewrite —
``catalog.move_head`` already enforces that; this module adds only the
receipt around it.
"""
from __future__ import annotations

import dataclasses
import json

from engine.v2.contracts import LegacyInputManifest, SnapshotImportRequest, TableContract
from engine.v2.data import documents, manifests
from engine.v2.data.catalog import move_head, record_failed_import
from engine.v2.data.errors import DataError
from engine.v2.data.import_snapshot import request_hash as compute_request_hash
from engine.v2.data.objects import FragmentInspection, partition_logical_hash, publish_legacy_file
from engine.v2.data.reference_catalog import insert_reference_inputs
from engine.v2.data.reference_inputs import publish_reference_inputs
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError, fail, make_problem
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.snapshot_import import protected_paths_hash
from engine.v2.ops.snapshots import commit_snapshot_for_attempt

__all__ = ["build_comparison_receipt", "legacy_rebuild_candidate_effect", "promote",
          "rollback", "snapshot_import_effect"]

#: Data-layer codes an ops caller can surface verbatim; anything else (say
#: ``MANIFEST_CORRUPT``, ``IDENTITY_CONFLICT``, ``SNAPSHOT_CONFLICT`` — none
#: registered in ``contracts.operations.FAILURE_CODES``) is reported as
#: ``VALIDATION_FAILED`` at the attempt level instead (this module's
#: judgement call; the ORIGINAL code is still the one persisted into the
#: failed ``SnapshotImportReceipt``, never lost).
_OPS_COMPATIBLE_CODES = frozenset({"INPUT_CHANGED", "RESOURCE_LIMIT_EXCEEDED"})


def _named(refs, name):
    for candidate_name, ref in refs:
        if candidate_name == name:
            return ref
    raise OpsError(make_problem("VALIDATION_FAILED", "required effect artifact is missing"))


def _translate(problem):
    code = problem.code if problem.code in _OPS_COMPATIBLE_CODES else "VALIDATION_FAILED"
    return make_problem(code, problem.message, details=problem.details)


def _find_ref(refs, path):
    for ref in refs:
        if ref.path == path:
            return ref
    raise fail("VALIDATION_FAILED", "inspection names a file its own plan never declared",
              details={"path": path})


# --------------------------------------------------------------------------
# snapshot_import
# --------------------------------------------------------------------------


def snapshot_import_effect(conn, store, claim, refs, *, clock):
    bindings = recorded_bindings(conn, claim.attempt_id)
    rows = [bindings.get(name) for name in ("snapshot_import_request.json",
                                            "legacy_table_mapping.json", "legacy_manifest.json")]
    if None in rows:
        raise OpsError(make_problem("VALIDATION_FAILED",
                                    "snapshot import request/mapping/manifest are not bound"))
    request_doc, mapping, manifest_doc = (
        json.loads(store.read_verified(artifact(conn, store, row.artifact_id))) for row in rows)
    request = from_document(SnapshotImportRequest, request_doc)
    manifest = from_document(LegacyInputManifest, manifest_doc)
    inspections = json.loads(store.read_verified(_named(refs, "snapshot_import")))
    req_hash = compute_request_hash(request)
    receipt_id = "recv_" + content_hash({"attempt_id": claim.attempt_id, "request_hash": req_hash}
                                        ).removeprefix("sha256:")[:32]
    try:
        receipt_ref = _commit_snapshot_import(conn, store, claim, (request, mapping, manifest),
                                              inspections, req_hash, receipt_id, clock=clock)
        return None, (("snapshot_import_receipt", receipt_ref),)
    except (DataError, OpsError) as exc:
        record_failed_import(conn, receipt_id=receipt_id, request_hash=req_hash,
                             attempt_id=claim.attempt_id, fence=claim.fence,
                             problem=exc.problem, clock=clock)
        raise OpsError(_translate(exc.problem)) from exc


def _commit_snapshot_import(conn, store, claim, documents_in, inspections, req_hash,
                            receipt_id, *, clock):
    request, mapping, manifest = documents_in
    if content_hash(to_document(manifest)) != request.source_manifest_hash:
        raise fail("VALIDATION_FAILED", "bound legacy manifest is not the request's own manifest")
    legacy_root = store.staging_dir(claim.attempt_id) / "legacy"
    contracts = {name: documents.decode_document(TableContract, doc)
                for name, doc in mapping["tables"].items()}
    all_objects, all_records, table_manifests = [], [], {}
    for table, partitions in inspections["tables"].items():
        contract, contract_ref = contracts[table], request.table_contract_refs[table]
        records, partition_hashes = [], {}
        for entry in partitions:
            partition_key, group_refs = entry["partition_key"], []
            for frag in entry["fragments"]:
                file_ref = _find_ref(request.table_sources[table], frag["path"])
                object_ref = publish_legacy_file(store, claim.attempt_id, legacy_root, file_ref)
                all_objects.append(object_ref)
                group_refs.append(object_ref)
                inspection = FragmentInspection(
                    object_ref=object_ref, partition_key=partition_key, row_count=frag["row_count"],
                    byte_hash=object_ref.content_hash, logical_content_hash=frag["logical_content_hash"],
                    primary_key_min=tuple(frag["primary_key_min"]),
                    primary_key_max=tuple(frag["primary_key_max"]),
                    time_min=frag["time_min"], time_max=frag["time_max"])
                records.append(manifests.fragment_record(
                    inspection, contract_ref, input_receipt_refs=(), import_request_hash=req_hash))
            if len(group_refs) > 1:
                partition_hashes[partition_key] = partition_logical_hash(
                    store, group_refs, contract, contract_ref, partition_key)
        table_manifests[table] = manifests.dataset_manifest(
            contract_ref, records, knowledge_mode=request.knowledge_mode_by_table[table],
            coverage_receipt_refs=(), availability_evidence_refs=(),
            partition_logical_hashes=partition_hashes)
        all_records.extend(records)

    calendar_version = "legacy_calendar:" + table_manifests["earnings_events"].logical_content_hash
    snapshot = manifests.snapshot_ref(
        table_manifests, calendar_version=calendar_version,
        source_priority_version=request.source_priority_version,
        finality_receipt_refs=request.finality_receipt_refs,
        parent_snapshot_id=request.expected_head_snapshot_id)

    legacy_snapshot_object_ref = publish_legacy_file(store, claim.attempt_id, legacy_root,
                                                      request.legacy_snapshot_source_ref)
    all_objects.append(legacy_snapshot_object_ref)
    references = publish_reference_inputs(store, claim.attempt_id, legacy_root, manifest.file_refs)

    receipt = commit_snapshot_for_attempt(
        conn, store, scope=request.scope, request_hash=req_hash, contracts=list(contracts.values()),
        objects=all_objects, records=all_records, manifests=list(table_manifests.values()),
        snapshot=snapshot, expected_head_snapshot_id=request.expected_head_snapshot_id,
        expected_head_generation=request.expected_head_generation, receipt_id=receipt_id,
        attempt_id=claim.attempt_id, fence=claim.fence, clock=clock,
        record_references=lambda c, rid: insert_reference_inputs(c, rid, references))
    receipt = dataclasses.replace(receipt, legacy_snapshot_object_ref=legacy_snapshot_object_ref)

    return store.publish_bytes(canonical_json(to_document(receipt)).encode("utf-8"),
                               schema_ref="snapshot_import_receipt.v1.0")


# --------------------------------------------------------------------------
# legacy_rebuild_candidate
# --------------------------------------------------------------------------


def legacy_rebuild_candidate_effect(conn, store, claim, refs, *, clock):
    """Refuse a candidate rebuild that touched anything outside its own root.

    The worker's own before/after report (``candidate_report.json``) is
    audit trail only — never trusted for the decision. This recomputes the
    current fingerprint of every declared protected path itself, with the
    SAME pure function the caller used to pin ``protected_before_hash``
    before submission, and compares directly (§10 point 1).
    """
    params = claim.spec.parameters
    protected = tuple(params.get("protected_paths", ()))
    pinned_before = params.get("protected_before_hash", "")
    current = protected_paths_hash(protected)
    if current != pinned_before:
        raise OpsError(make_problem(
            "VALIDATION_FAILED", "legacy rebuild candidate changed a protected path",
            details={"protected_paths": list(protected)}))
    return None, ()


# --------------------------------------------------------------------------
# comparison, promotion, rollback
# --------------------------------------------------------------------------


def _scope_head(conn, scope):
    row = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                       (scope,)).fetchone()
    return (row["snapshot_id"], row["generation"]) if row is not None else (None, 0)


def _table_summary(conn, dvr):
    row = conn.execute("SELECT row_count, logical_content_hash FROM data_dataset_versions "
                       "WHERE dataset_version_id=?", (dvr.dataset_version_id,)).fetchone()
    return {"contract_id": dvr.table_contract_ref.contract_id,
           "definition_hash": dvr.table_contract_ref.definition_hash,
           "dataset_version_id": dvr.dataset_version_id,
           "row_count": row["row_count"], "logical_content_hash": row["logical_content_hash"]}


def _publish_receipt(store, conn, document, *, clock):
    ref = store.publish_bytes(canonical_json(document).encode("utf-8"),
                              schema_ref="snapshot_update_receipt.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def build_comparison_receipt(conn, store, *, candidate_scope, target_scope, clock):
    """Per-table dataset logical hash and row count, candidate vs. target head.

    Required by :func:`promote` before it will move any head (§10 point 4).
    """
    candidate_id, _ = _scope_head(conn, candidate_scope)
    if candidate_id is None:
        raise fail("SNAPSHOT_NOT_READY", "candidate scope has no committed head",
                  details={"scope": candidate_scope})
    target_id, _ = _scope_head(conn, target_scope)
    candidate_snap = Repository(conn).resolve(candidate_id)
    target_snap = Repository(conn).resolve(target_id) if target_id is not None else None

    tables, mismatches = {}, []
    for name, dvr in candidate_snap.table_versions.items():
        candidate_entry = _table_summary(conn, dvr)
        target_entry = None
        if target_snap is not None and name in target_snap.table_versions:
            target_entry = _table_summary(conn, target_snap.table_versions[name])
            if (target_entry["contract_id"] != candidate_entry["contract_id"]
                    or target_entry["definition_hash"] != candidate_entry["definition_hash"]):
                mismatches.append(name)
        tables[name] = {"candidate": candidate_entry, "target": target_entry}

    document = {"schema_version": "snapshot_comparison_receipt.v1.0",
               "candidate_scope": candidate_scope, "target_scope": target_scope,
               "candidate_snapshot_id": candidate_id, "target_snapshot_id": target_id,
               "tables": tables, "contract_mismatches": mismatches,
               "generated_at": format_timestamp(clock.now())}
    ref = store.publish_bytes(canonical_json(document).encode("utf-8"),
                              schema_ref="snapshot_comparison_receipt.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def promote(conn, store, *, candidate_scope, target_scope, expected_snapshot_id, expected_generation,
           comparison_receipt_id, clock):
    """Explicit promotion (§10 point 4): never a side effect of worker success."""
    document = json.loads(store.read_verified(artifact(conn, store, comparison_receipt_id)))
    if (document.get("candidate_scope") != candidate_scope
            or document.get("target_scope") != target_scope):
        raise fail("VALIDATION_FAILED", "comparison receipt does not match this promotion",
                  details={"comparison_receipt_id": comparison_receipt_id})
    if document.get("contract_mismatches"):
        # CONTRACT_MISMATCH is a data-layer code (contracts.data.DATA_FAILURE_CODES);
        # ops.errors.FAILURE_CODES has no entry for it, so this ops-layer
        # refusal is reported as VALIDATION_FAILED with the mismatch detail
        # preserved (this module's judgement call).
        raise fail("VALIDATION_FAILED", "candidate and target disagree on a table contract",
                  details={"tables": document["contract_mismatches"]})
    current_candidate_id, _ = _scope_head(conn, candidate_scope)
    if current_candidate_id != document.get("candidate_snapshot_id"):
        raise fail("STALE_EXPECTATION", "comparison receipt is stale; the candidate head has moved",
                  details={"candidate_scope": candidate_scope})
    to_snapshot_id = document["candidate_snapshot_id"]
    receipt_ref = _publish_receipt(store, conn, {
        "schema_version": "snapshot_update_receipt.v1.0", "action": "promote", "scope": target_scope,
        "from_snapshot_id": expected_snapshot_id, "to_snapshot_id": to_snapshot_id,
        "comparison_receipt_ref": comparison_receipt_id, "at": format_timestamp(clock.now())},
        clock=clock)
    move_head(conn, scope=target_scope, to_snapshot_id=to_snapshot_id,
             expected_snapshot_id=expected_snapshot_id, expected_generation=expected_generation,
             receipt_ref=receipt_ref.artifact_id, clock=clock)
    return receipt_ref


def rollback(conn, store, *, scope, to_snapshot_id, expected_snapshot_id, expected_generation, clock):
    """Compare-and-swap ``scope``'s head to a prior committed snapshot (§10 point 5).

    Never deletes, rewrites or relabels either snapshot: ``Repository.resolve``
    proves ``to_snapshot_id`` is still a genuine, resolvable committed
    snapshot before the swap, and ``catalog.move_head`` itself never mints
    or removes a row — both stay resolvable afterward.
    """
    Repository(conn).resolve(to_snapshot_id)
    receipt_ref = _publish_receipt(store, conn, {
        "schema_version": "snapshot_update_receipt.v1.0", "action": "rollback", "scope": scope,
        "from_snapshot_id": expected_snapshot_id, "to_snapshot_id": to_snapshot_id,
        "at": format_timestamp(clock.now())}, clock=clock)
    move_head(conn, scope=scope, to_snapshot_id=to_snapshot_id, expected_snapshot_id=expected_snapshot_id,
             expected_generation=expected_generation, receipt_ref=receipt_ref.artifact_id, clock=clock)
    return receipt_ref
