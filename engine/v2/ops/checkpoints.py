"""Coordinator-only checkpoint publication and verified reuse."""
from __future__ import annotations

from engine.v2.contracts import ArtifactRef, CheckpointReceipt
from engine.v2.foundation import ArtifactStore, content_hash, format_timestamp
from engine.v2.ops.catalog import dumps, load_json, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.lifecycle import verify_fence


def cache_identity(*, kind, inputs, implementation, parameters, environment, schema, shard):
    return content_hash(dict(kind=kind, inputs=inputs, implementation=implementation,
                             parameters=parameters, environment=environment,
                             schema=schema, shard=shard))


def register_artifact(conn, ref, attempt_id, clock):
    old = conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?",
                       (ref.artifact_id,)).fetchone()
    if old:
        if old[0] != dumps(ref):
            raise fail("INTEGRITY_FAILED", "artifact identity conflict")
        return
    conn.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?)",
                 (ref.artifact_id, dumps(ref), attempt_id, format_timestamp(clock.now())))


def artifact(conn, store, artifact_id):
    row = conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?",
                       (artifact_id,)).fetchone()
    if row is None:
        raise fail("INTEGRITY_FAILED", "required artifact is missing")
    ref = load_json(ArtifactRef, row[0])
    store.verify(ref)
    return ref


def commit_checkpoint(conn, store: ArtifactStore, claim, candidate, *, clock, fault=None):
    expected = cache_identity(
        kind=claim.spec.kind, inputs=content_hash(list(claim.spec.input_refs)),
        implementation=claim.spec.implementation_ref, parameters=content_hash(claim.spec.parameters),
        environment=claim.spec.environment_ref, schema=candidate.output_schema_ref,
        shard=candidate.shard_key)
    identities = (candidate.input_hash, candidate.implementation_hash,
                  candidate.parameter_hash, candidate.environment_hash)
    required = (content_hash(list(claim.spec.input_refs)), claim.spec.implementation_ref,
                content_hash(claim.spec.parameters), claim.spec.environment_ref)
    if candidate.cache_key != expected or identities != required or not candidate.outputs:
        raise fail("CHECKPOINT_INCOMPATIBLE", "checkpoint does not match the admitted inputs")
    refs = tuple(store.publish_candidate(claim.attempt_id, output.staged_path,
                                         schema_ref=output.schema_ref,
                                         max_bytes=claim.resources.scratch_limit_bytes)
                 for output in candidate.outputs)
    receipt = CheckpointReceipt(
        stage_id=claim.spec.kind, shard_key=candidate.shard_key, cache_key=expected,
        input_hash=required[0], implementation_hash=required[1], parameter_hash=required[2],
        environment_hash=required[3], output_schema_ref=candidate.output_schema_ref,
        artifact_refs=refs, validation_refs=(), producer_attempt_id=claim.attempt_id,
        producer_fence=claim.fence, committed_at=format_timestamp(clock.now()))
    if fault:
        fault("before_catalog")
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        old = conn.execute("SELECT receipt_json FROM checkpoints WHERE cache_key = ?", (expected,)).fetchone()
        if old:
            previous = load_json(CheckpointReceipt, old[0])
            if previous.artifact_refs != refs:
                raise fail("INTEGRITY_FAILED", "deterministic checkpoint changed content")
            return previous
        for ref in refs:
            register_artifact(conn, ref, claim.attempt_id, clock)
        conn.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                     (expected, receipt.stage_id, receipt.shard_key, dumps(receipt), claim.attempt_id))
    if fault:
        fault("after_catalog")
    return receipt


def reuse(conn, store, cache_key):
    row = conn.execute("SELECT receipt_json FROM checkpoints WHERE cache_key = ?", (cache_key,)).fetchone()
    if not row:
        return None
    receipt = load_json(CheckpointReceipt, row[0])
    for ref in receipt.artifact_refs:
        store.verify(ref)
    return receipt


def assemble(expected_ids, batches, *, no_work_receipt=None):
    """Population validation precedes concatenation, including legitimate refusals."""
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise fail("VALIDATION_FAILED", "duplicate expected population", details={"field": "expected_ids"})
    rows = [row for batch in batches for row in batch]
    actual = [row["row_id"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise fail("VALIDATION_FAILED", "batch coverage differs", details={
            "field": "row_id", "missing": sorted(set(expected) - set(actual)),
            "duplicate_count": len(actual) - len(set(actual))})
    if not expected and no_work_receipt != {"expected_count": 0, "reason": "no_eligible_inputs"}:
        raise fail("VALIDATION_FAILED", "empty work requires a positive no-work receipt")
    by_id = {row["row_id"]: row for row in rows}
    return [by_id[key] for key in expected]
