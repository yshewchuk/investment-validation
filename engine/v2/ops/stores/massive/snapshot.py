"""Supervised network-free Massive bootstrap or explicitly parent-bound refresh."""
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from engine.v2.data import (
    errors,
    manifests,
    massive_actions,
    massive_daily,
    massive_refresh,
    objects,
    repository,
)
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    from_document,
)
from engine.v2.ops import catalog, lifecycle, snapshots, submission
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.stores.massive import pinned

KIND = "massive_daily_snapshot"
SCHEMA = snapshots.SNAPSHOT_REF_SCHEMA_REF


@dataclass(frozen=True)
class Parameters:
    acquisition_job: str
    input_bindings: dict[str, str]
    expected_ids: tuple[str, ...]
    parent_snapshot_id: str | None = None
    scope: str | None = None
    head_generation: int | None = None


def _parent_problems(params):
    parent_fields = (params.parent_snapshot_id, params.scope, params.head_generation)
    if any(v is not None for v in parent_fields) and (
            not isinstance(params.parent_snapshot_id, str)
            or re.fullmatch(r"snap_[0-9a-f]{32}", params.parent_snapshot_id) is None
            or not isinstance(params.scope, str) or not 1 <= len(params.scope) <= 128
            or params.scope.strip() != params.scope
            or type(params.head_generation) is not int or params.head_generation < 1):
        return ("Massive refresh requires an explicit parent, scope and positive head generation",)
    return ()


def parameter_problems(job, params):
    if (re.fullmatch(r"job_[0-9a-f]{32}", params.acquisition_job) is None
            or params.input_bindings != {"acquisition.json": params.acquisition_job + "#massive_acquisition"}
            or params.expected_ids != (params.acquisition_job,) or job.input_refs
            or job.dependency_job_ids != (params.acquisition_job,) or job.provider_budget_ref is not None):
        return ("Massive snapshot requires one declared acquisition dependency and no provider account",)
    return _parent_problems(params)


def job_kind():
    return submission.JobKind(name=KIND, worker=KIND, parameters=Parameters,
        resource_classes=frozenset({"io_fetch"}), effects=("staged",),
        retry=submission.RetryPolicy("bounded", 3, (1, 2)), checkpoint_contract=SCHEMA,
        namespaces=frozenset({"shadow"}), validate=parameter_problems)


def _existing(conn, store, identity, artifact_id, contract=massive_daily.CONTRACT, *, parent=None, scope=None):
    row = conn.execute("SELECT * FROM data_import_receipts WHERE receipt_id=?", (identity,)).fetchone()
    if row is None:
        return None
    if (row["scope"], row["source_manifest_hash"], row["status"]) != (scope or identity, identity, "committed"):
        raise fail("INTEGRITY_FAILED", "Massive snapshot receipt identity differs")
    resolved = repository.Repository(conn, store).resolve_full(row["result_snapshot_id"])
    _verify_output(resolved, artifact_id, contract, parent)
    target = {f.fragment_id for f in resolved.table_manifests[contract.table_name].fragment_refs}
    for record in resolved.records:
        if record.fragment_id in target:
            objects.verify_object_path(store, record.object_ref)
    return resolved.snapshot


def _verify_output(resolved, artifact_id, contract, parent):
    prior = parent.table_manifests.get(contract.table_name) if parent else None
    pins = tuple(sorted(set((*prior.coverage_receipt_refs, artifact_id)))) if prior else (artifact_id,)
    unchanged = {k: v for k, v in parent.table_manifests.items() if k != contract.table_name} if parent else {}
    target = resolved.table_manifests[contract.table_name]
    if (contract not in resolved.contracts
            or {k: v for k, v in resolved.table_manifests.items() if k != contract.table_name} != unchanged
            or resolved.table_manifests[contract.table_name].coverage_receipt_refs != pins
            or (target.knowledge_mode, target.availability_evidence_refs) != ("reconstructed", ())
            or (parent is None and resolved.snapshot.parent_snapshot_id is not None)):
        raise fail("INTEGRITY_FAILED", "Massive snapshot receipt output differs")
    if parent and any(getattr(resolved.snapshot, k) != getattr(parent.snapshot, k)
                      for k in ("calendar_version", "source_priority_version", "finality_receipt_refs")):
        raise fail("INTEGRITY_FAILED", "Massive snapshot parent policies differ")


def _publish(conn, store, params, envelope, clock, fault):
    def fence():
        with catalog.transaction(conn):
            lifecycle.verify_fence(conn, envelope["attempt_id"], envelope["fence"], clock.now())
    fence()
    artifact_id, plan = pinned.acquisition_input(conn, store, envelope["attempt_id"], params.acquisition_job)
    dataset = plan["request"]["dataset"]
    contract = massive_daily.CONTRACT if dataset == "daily" else massive_actions.CONTRACTS[dataset]
    normalizer = massive_daily.NORMALIZER if dataset == "daily" else massive_actions.NORMALIZER
    parent = repository.Repository(conn, store).resolve_full(params.parent_snapshot_id) if params.parent_snapshot_id else None
    identity_parts = [artifact_id, normalizer, contract.definition_hash]
    if parent:
        identity_parts += ["massive_interval_replace.v1", params.parent_snapshot_id, params.scope, params.head_generation]
    identity = content_hash(identity_parts)
    def replay():
        return _existing(conn, store, identity, artifact_id, contract, parent=parent, scope=params.scope)
    existing = replay()
    if existing is not None:
        fence()
        return existing
    try:
        rows = pinned.normalized_rows(conn, store, plan)
    except OpsError as error:
        if error.code == "INVALID_REQUEST":
            raise fail("INTEGRITY_FAILED", "Massive saved evidence scope is invalid") from None
        raise
    if parent:
        built = massive_refresh.build_candidate(conn, store, parent, contract, plan["request"], rows, artifact_id, identity)
    else:
        record, manifest, candidate = massive_daily.build_candidate(store, rows, artifact_id, identity, contract=contract)
        built = manifests.ResolvedSnapshot(contracts=(contract,), objects=(record.object_ref,), records=(record,),
                                          table_manifests={contract.table_name: manifest}, snapshot=candidate)
    try:
        receipt = snapshots.commit_snapshot_for_attempt(conn, store, scope=params.scope or identity, request_hash=identity,
            contracts=built.contracts, objects=built.objects, records=built.records,
            manifests=tuple(built.table_manifests.values()), snapshot=built.snapshot,
            expected_head_snapshot_id=params.parent_snapshot_id,
            expected_head_generation=params.head_generation or 0, receipt_id=identity, attempt_id=envelope["attempt_id"],
            fence=envelope["fence"], clock=clock, fault=fault)
        return receipt.snapshot_ref
    except errors.DataError as error:
        if error.code in ("IDENTITY_CONFLICT", "SNAPSHOT_CONFLICT"):
            existing = replay()
            if existing is not None:
                fence()
                return existing
        raise


def run(parameters, root, envelope, *, clock=None, fault=None):
    try:
        params = from_document(Parameters, parameters)
    except DocumentError:
        raise fail("INVALID_REQUEST", "Massive snapshot parameters are invalid") from None
    if _parent_problems(params):
        raise fail("INVALID_REQUEST", "Massive refresh parent parameters are invalid")
    if not all(envelope.get(k) for k in ("attempt_id", "fence", "catalog_path", "objects_root")):
        raise fail("INVALID_REQUEST", "Massive snapshot requires trusted launch context")
    try:
        with closing(catalog.connect(envelope["catalog_path"], must_exist=True)) as conn:
            result = _publish(conn, ArtifactStore(envelope["objects_root"]), params, envelope,
                              clock or SystemClock(), fault)
        Path(root, "snapshot.json").write_text(catalog.dumps(result))
    except errors.DataError as error:
        code = {"SNAPSHOT_CONFLICT": "RESOURCE_UNAVAILABLE", "RESULT_LIMIT_EXCEEDED": "RESOURCE_LIMIT_EXCEEDED"}
        raise fail(code.get(error.code, "INTEGRITY_FAILED"),
                   "Massive snapshot data is unavailable or invalid") from None
    except (ArtifactError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        raise fail("INTEGRITY_FAILED", "Massive snapshot evidence is invalid") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "Massive snapshot storage is unavailable") from None
    return {"outputs": [{"name": "snapshot", "path": "snapshot.json", "schema": SCHEMA}],
            "completed_ids": list(params.expected_ids), "no_work": False}
