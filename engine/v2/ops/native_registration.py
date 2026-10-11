"""Verify native registration evidence owned by the existing catalog reservation."""
import json

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
)
from engine.v2.ops.errors import fail


def verified_registration_bytes(conn, store, run_id, *, expected_ref=None):
    """Return exact verified registration bytes, or None for an absent reservation."""
    row = conn.execute(
        "SELECT e.*, h.input_hash AS registered_input, h.payload_hash AS registered_payload, "
        "h.payload_json AS registered_document FROM experiment_runs e "
        "LEFT JOIN hypotheses h ON h.run_id=e.run_id AND h.spec_hash=e.spec_hash "
        "WHERE e.run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    try:
        evidence = json.loads(row["evidence_json"])
        ref = from_document(ArtifactRef, evidence["native_registration"])
        if (ref.schema_ref not in ("native_experiment_registration.v1.0", "native_experiment_registration.v1.1")
                or (expected_ref is not None and ref != expected_ref)):
            raise ValueError
        binding = store.read_verified(ref)
        document = json.loads(binding)
        valid = _matches(row, evidence, ref, binding, document)
    except (ArtifactError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding")
    return binding


def _matches(row, evidence, ref, binding, document):
    variant = content_hash(document)
    return (document["schema_version"] == ref.schema_ref
            and canonical_json(document).encode() == binding
            and artifact_reference(binding, ref.schema_ref) == ref
            and row["spec_hash"] == document["spec_hash"] and row["mode"] == "primary"
            and row["input_hash"] == row["registered_input"] == variant
            and row["registered_payload"] == content_hash({"spec": row["spec_hash"], "input": variant})
            and row["registered_document"] == "{}" and evidence["variant_id"] == variant
            and type(evidence["planned_variants"]) is int and evidence["planned_variants"] == 1)
