"""Supervised, network-free bootstrap of one pinned Massive daily snapshot."""
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from engine.v2.data import errors, massive_daily, objects, repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    from_document,
)
from engine.v2.ops import catalog, lifecycle, snapshots, submission
from engine.v2.ops.errors import fail
from engine.v2.ops.stores.massive import pinned

KIND = "massive_daily_snapshot"
SCHEMA = snapshots.SNAPSHOT_REF_SCHEMA_REF


@dataclass(frozen=True)
class Parameters:
    acquisition_job: str
    input_bindings: dict[str, str]
    expected_ids: tuple[str, ...]


def parameter_problems(job, params):
    if (re.fullmatch(r"job_[0-9a-f]{32}", params.acquisition_job) is None
            or params.input_bindings != {"acquisition.json": params.acquisition_job + "#massive_acquisition"}
            or params.expected_ids != (params.acquisition_job,) or job.input_refs
            or job.dependency_job_ids != (params.acquisition_job,) or job.provider_budget_ref is not None):
        return ("Massive snapshot requires one declared acquisition dependency and no provider account",)
    return ()


def job_kind():
    return submission.JobKind(name=KIND, worker=KIND, parameters=Parameters,
        resource_classes=frozenset({"io_fetch"}), effects=("staged",),
        retry=submission.RetryPolicy("bounded", 3, (1, 2)), checkpoint_contract=SCHEMA,
        namespaces=frozenset({"shadow"}), validate=parameter_problems)


def _existing(conn, store, identity, artifact_id):
    row = conn.execute("SELECT * FROM data_import_receipts WHERE receipt_id=?", (identity,)).fetchone()
    if row is None:
        return None
    if (row["scope"], row["source_manifest_hash"], row["status"]) != (identity, identity, "committed"):
        raise fail("INTEGRITY_FAILED", "Massive snapshot receipt identity differs")
    resolved = repository.Repository(conn, store).resolve_full(row["result_snapshot_id"])
    if (resolved.contracts != (massive_daily.CONTRACT,) or resolved.snapshot.parent_snapshot_id is not None
            or resolved.table_manifests[massive_daily.TABLE].coverage_receipt_refs != (artifact_id,)):
        raise fail("INTEGRITY_FAILED", "Massive snapshot receipt output differs")
    for obj in resolved.objects:
        objects.verify_object_path(store, obj)
    return resolved.snapshot


def _publish(conn, store, params, envelope, clock, fault):
    def fence():
        with catalog.transaction(conn):
            lifecycle.verify_fence(conn, envelope["attempt_id"], envelope["fence"], clock.now())
    fence()
    artifact_id, plan = pinned.acquisition_input(conn, store, envelope["attempt_id"], params.acquisition_job)
    identity = content_hash([artifact_id, massive_daily.NORMALIZER, massive_daily.CONTRACT.definition_hash])
    existing = _existing(conn, store, identity, artifact_id)
    if existing is not None:
        fence()
        return existing
    rows = pinned.normalized_rows(conn, store, plan)
    record, manifest, candidate = massive_daily.build_candidate(store, rows, artifact_id, identity)
    try:
        receipt = snapshots.commit_snapshot_for_attempt(conn, store, scope=identity, request_hash=identity,
            contracts=(massive_daily.CONTRACT,), objects=(record.object_ref,), records=(record,),
            manifests=(manifest,), snapshot=candidate, expected_head_snapshot_id=None,
            expected_head_generation=0, receipt_id=identity, attempt_id=envelope["attempt_id"],
            fence=envelope["fence"], clock=clock, fault=fault)
        return receipt.snapshot_ref
    except errors.DataError as error:
        if error.code in ("IDENTITY_CONFLICT", "SNAPSHOT_CONFLICT"):
            existing = _existing(conn, store, identity, artifact_id)
            if existing is not None:
                fence()
                return existing
        raise


def run(parameters, root, envelope, *, clock=None, fault=None):
    try:
        params = from_document(Parameters, parameters)
    except DocumentError:
        raise fail("INVALID_REQUEST", "Massive snapshot parameters are invalid") from None
    if not all(envelope.get(k) for k in ("attempt_id", "fence", "catalog_path", "objects_root")):
        raise fail("INVALID_REQUEST", "Massive snapshot requires trusted launch context")
    try:
        with closing(catalog.connect(envelope["catalog_path"], must_exist=True)) as conn:
            result = _publish(conn, ArtifactStore(envelope["objects_root"]), params, envelope,
                              clock or SystemClock(), fault)
        Path(root, "snapshot.json").write_text(catalog.dumps(result))
    except errors.DataError as error:
        raise fail("RESOURCE_UNAVAILABLE" if error.code == "SNAPSHOT_CONFLICT" else "INTEGRITY_FAILED",
                   "Massive snapshot data is unavailable or invalid") from None
    except (ArtifactError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        raise fail("INTEGRITY_FAILED", "Massive snapshot evidence is invalid") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "Massive snapshot storage is unavailable") from None
    return {"outputs": [{"name": "snapshot", "path": "snapshot.json", "schema": SCHEMA}],
            "completed_ids": list(params.expected_ids), "no_work": False}
