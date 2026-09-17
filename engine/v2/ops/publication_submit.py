"""Submit a fresh, fenced publication from verified retained inputs.

This is deliberately a submission helper, not a publisher.  The returned
job is claimed and committed by :class:`supervisor.Service`, which remains
the only path that calls ``publication_effect`` and advances ``CURRENT``.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.submission import NamespacePolicy, get_job, submit

__all__ = ["submit_retained_publication"]


_REQUIRED = ("bundle.tar", "finality.json", "selfcheck.json", "engineering_gate.json")


def _source_attempt(conn, job_id):
    row = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "source publication has no completed attempt",
                   details={"source_job_id": job_id})
    return row[0]


def submit_retained_publication(conn, store, *, registry, policy: NamespacePolicy, clock,
                                source_job_id: str, projection_binding_ref: str,
                                operation_id: str, principal="operator"):
    """Submit an idempotent fresh publication from a completed source attempt.

    ``operation_id`` is operator-chosen and becomes an immutable bound
    artifact.  Repeating it retries the identical job; a rollback uses a new
    operation id, making a distinct generation even when it reuses projection
    A after projection B.
    """
    source = get_job(conn, source_job_id)
    if source.kind != "publication" or source.state != "succeeded":
        raise fail("INPUT_CHANGED", "source job is not a completed publication",
                   details={"source_job_id": source_job_id})
    attempt_id = _source_attempt(conn, source_job_id)
    retained = recorded_bindings(conn, attempt_id)
    missing = [name for name in _REQUIRED if name not in retained]
    if missing:
        raise fail("INPUT_CHANGED", "source publication lacks retained bindings",
                   details={"missing": missing})
    projection = artifact(conn, store, projection_binding_ref)
    operation = store.publish_bytes(json.dumps({
        "schema_version": "retained_publication_operation.v1.0",
        "source_publication_job_id": source_job_id,
        "projection_binding_ref": projection.artifact_id,
        "operation_id": operation_id,
    }, sort_keys=True).encode(), schema_ref="retained_publication_operation.v1.0")
    with transaction(conn):
        register_artifact(conn, operation, None, clock)

    bindings = {name: item.artifact_id for name, item in retained.items()}
    bindings["projection_binding.json"] = projection.artifact_id
    bindings["publication_operation.json"] = operation.artifact_id
    source_spec = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (source_job_id,)).fetchone()
    if source_spec is None:
        raise fail("INPUT_CHANGED", "source publication disappeared",
                   details={"source_job_id": source_job_id})
    spec_doc = json.loads(source_spec["spec_json"])
    parameters = dict(spec_doc["parameters"])
    parameters["input_bindings"] = bindings
    refs = tuple(sorted(set(spec_doc["input_refs"]) | {item.artifact_id for item in retained.values()} |
                        {projection.artifact_id, operation.artifact_id}))
    key = "retained-publication:" + content_hash({
        "source": source_job_id, "projection": projection.artifact_id, "operation": operation_id,
    }).split(":", 1)[1][:32]
    profile = profile_named(DEFAULT_POLICY, "delivery")
    threads = profile.thread_count or profile.cpu_count
    job = JobSpec(kind="publication",
                  implementation_ref=content_hash(worker_source_manifest(Path(__file__).resolve().parents[3])),
                  spec_hash=None, environment_ref=content_hash(environment_identity(threads)), parameters=parameters,
                  input_refs=refs, dependency_job_ids=(source_job_id,), output_namespace=source.namespace,
                  resource_class="delivery", retry_policy_ref="bounded",
                  checkpoint_contract_ref="effect_receipt.v1.0")
    return submit(conn, registry, policy, SubmitRequest(namespace=source.namespace,
                  idempotency_key=key, principal=principal, job=job), clock=clock)
