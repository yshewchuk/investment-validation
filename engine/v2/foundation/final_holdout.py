"""Pure final-read metadata validation, never authorization or data access."""
import re

from engine.v2.contracts import ArtifactRef, SnapshotRef
from engine.v2.contracts.final_holdout import FinalHoldoutDeclaration, FinalHoldoutRequest
from engine.v2.foundation.artifacts import artifact_reference
from engine.v2.foundation.canonical import canonical_json, content_hash
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.foundation.typed import from_document, parse_schema_version, to_document


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("invalid final-read content hash")


def _reference(ref, schema=None):
    _hash(ref.content_hash)
    parse_schema_version(ref.schema_ref)
    digest = ref.content_hash.removeprefix("sha256:")
    identity = content_hash({"content_hash": ref.content_hash, "schema_ref": ref.schema_ref})
    if (ref.byte_size <= 0 or ref.storage_key != f"objects/{digest[:2]}/{digest}"
            or ref.artifact_id != "art_" + identity.removeprefix("sha256:")[:32]
            or (schema is not None and ref.schema_ref != schema)):
        raise ValueError("invalid final-read artifact reference")


def declaration_from_document(document):
    """Decode references sealed by native preregistration, without opening them."""
    value = from_document(FinalHoldoutDeclaration, document)
    _reference(value.evaluation_recipe_ref, "final_holdout_recipe.v1.0")
    _reference(value.completeness_bar_ref, "final_holdout_completeness.v1.0")
    for ref in value.frozen_model_refs:
        _reference(ref)
    return value


def request_from_document(document):
    """Strict request shape; valid metadata is not proof of read eligibility."""
    value = from_document(FinalHoldoutRequest, document)
    for text in (value.experiment_id, value.registration_run_id, value.snapshot_id):
        if not text or text != text.strip():
            raise ValueError("invalid final-read identity")
    for digest in (value.variant_id, value.execution_plan_hash, value.source_closure_hash):
        _hash(digest)
    _reference(value.registration_ref, "native_experiment_registration.v1.1")
    if value.variant_id != value.registration_ref.content_hash:
        raise ValueError("final-read variant differs from registration")
    membership = ExperimentHoldouts(value.holdout_as_of_month)
    if (value.random_membership_version != membership.random_version
            or value.rolling_membership_version != membership.rolling_version):
        raise ValueError("unsupported final-read membership version")
    declaration_from_document(to_document(value.final_read))
    return value


def request_from_registration(document, run_id, ref):
    """Project already-verified canonical registration metadata, never new pins."""
    try:
        return _project_registration(document, run_id, ref)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid final-read registration metadata") from error


def _project_registration(document, run_id, ref):
    if document["schema_version"] != "native_experiment_registration.v1.1":
        raise ValueError("registration has no final-read declaration")
    expected = artifact_reference(canonical_json(document).encode(), document["schema_version"])
    ref = from_document(ArtifactRef, to_document(ref))
    if ref != expected:
        raise ValueError("registration reference differs from metadata")
    plan = document["execution_plan"]
    if (plan["schema_version"] != "experiment_execution_plan.v1.0"
            or not isinstance(plan["arms"], list) or len(plan["arms"]) != 1
            or not isinstance(plan["arms"][0], str) or not plan["arms"][0].strip()):
        raise ValueError("final read requires one registered arm")
    snapshot = from_document(SnapshotRef, document["snapshot"])
    _hash(snapshot.manifest_hash)
    sources = document["source_closure"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("final read requires registered source identity")
    for digest in sources.values():
        _hash(digest)
    value = FinalHoldoutRequest(
        experiment_id=plan["experiment_id"], registration_run_id=run_id,
        variant_id=ref.content_hash, registration_ref=ref, snapshot_id=snapshot.snapshot_id,
        execution_plan_hash=content_hash(plan), source_closure_hash=content_hash(sources),
        final_read=declaration_from_document(document["final_read"]), **document["holdouts"])
    return request_from_document(to_document(value))
