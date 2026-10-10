"""Exact native preregistration over the existing experiment catalog records."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    safe_relative_path,
    to_document,
)
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import (
    ExperimentSpec,
    register_hypothesis_in_transaction,
    resolve_experiment_plan,
)
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research.experiment_population import load_population


@dataclass(frozen=True)
class NativeRegistration:
    """Immutable canonical binding; each document access returns a fresh copy."""
    run_id: str
    binding_json: bytes

    @property
    def document(self):
        return json.loads(self.binding_json)

    @property
    def variant_id(self):
        return content_hash(self.document)


def _binding(repository, spec, *, code_root, as_of_month, scope, event_ids, expected=None):
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "native registration requires an experiment specification")
    plan = resolve_experiment_plan(spec)
    try:
        safe_relative_path(plan.runner)
        if (spec.input_files != () or not plan.runner.endswith(".py")
                or not isinstance(spec.hypothesis, str) or not spec.hypothesis.strip()
                or not plan.arms[0].strip() or not plan.experiment_id.strip()):
            raise ValueError
    except (ArtifactError, ValueError):
        raise fail("INVALID_EXPERIMENT_SPEC", "native registration requires a Python runner and no file inputs") from None
    spec = replace(spec, economic_params=plan.as_document()["economic_params"], input_files=())
    try:
        root = Path(code_root).resolve()
        if (root / plan.runner).resolve() != root / plan.runner:
            raise ValueError
        sources = source_closure(root, [plan.runner])
        if any((root / name).resolve() != root / name for name in sources):
            raise ValueError
    except (ArtifactError, OpsError, OSError, SyntaxError, ValueError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "native source closure is missing, malformed or indirect") from None
    try:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError
        snapshot = repository.resolve_pinned(scope)
    except (DataError, ValueError, TypeError, KeyError):
        raise fail("SNAPSHOT_UNRESOLVED", "native registration requires a current committed snapshot") from None
    document = {
        "schema_version": "native_experiment_registration.v1.0",
        "spec_hash": spec.spec_hash, "execution_plan": plan.as_document(),
        "snapshot": to_document(snapshot), "scope": scope,
        "source_closure": sources, "environment": environment_identity(1),
    }
    if expected is not None and any(expected.document[name] != value for name, value in document.items()):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native execution inputs changed before population admission")
    events = load_population(repository, snapshot, as_of_month=as_of_month,
                             purpose="selection", event_ids=event_ids)
    document.update({
        "event_ids": sorted(events["event_id"]),
        "holdouts": {name: events.iloc[0][name] for name in (
            "holdout_as_of_month", "random_membership_version", "rolling_membership_version")},
    })
    key = content_hash({"experiment_id": plan.experiment_id, "arm": plan.arms[0]})
    return NativeRegistration("native_" + key.removeprefix("sha256:"),
                              canonical_json(document).encode("utf-8")), spec


def _existing(conn, store, registration):
    row = conn.execute(
        "SELECT e.*, h.input_hash AS registered_input, h.payload_hash AS registered_payload, "
        "h.payload_json AS registered_document FROM experiment_runs e "
        "LEFT JOIN hypotheses h ON h.run_id=e.run_id AND h.spec_hash=e.spec_hash "
        "WHERE e.run_id=?", (registration.run_id,)).fetchone()
    if row is None:
        return False
    try:
        evidence = json.loads(row["evidence_json"])
        ref = from_document(ArtifactRef, evidence["native_registration"])
        expected_ref = artifact_reference(registration.binding_json, "native_experiment_registration.v1.0")
        valid = (row["spec_hash"] == registration.document["spec_hash"]
                 and row["mode"] == "primary"
                 and row["input_hash"] == row["registered_input"] == registration.variant_id
                 and row["registered_payload"] == content_hash({
                     "spec": row["spec_hash"], "input": registration.variant_id})
                 and row["registered_document"] == "{}" and ref == expected_ref
                 and evidence["variant_id"] == registration.variant_id
                 and type(evidence["planned_variants"]) is int and evidence["planned_variants"] == 1
                 and store.read_verified(ref) == registration.binding_json)
    except (ArtifactError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding")
    return True


def register_native(conn, store, repository, spec, *, code_root, as_of_month,
                    scope="shadow", event_ids=None):
    """Explicitly reserve an exact current binding before any outcome is read.

    Publication precedes the short catalog transaction. A failed commit can
    leave an unreferenced complete object, never a partially admitted binding.
    """
    registration, spec = _binding(repository, spec, code_root=code_root, as_of_month=as_of_month,
                                  scope=scope, event_ids=event_ids)
    if _existing(conn, store, registration):
        return registration
    ref = store.publish_bytes(registration.binding_json,
                              schema_ref="native_experiment_registration.v1.0")
    with transaction(conn):
        if _existing(conn, store, registration):
            return registration
        if conn.execute("SELECT 1 FROM hypotheses WHERE spec_hash=?", (spec.spec_hash,)).fetchone():
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native specification is already reserved")
        run_id, created = register_hypothesis_in_transaction(
            conn, spec, registration.variant_id, mode="primary", run_id=registration.run_id)
        if not created or run_id != registration.run_id:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration key is already reserved")
        evidence = {"native_registration": to_document(ref),
                    "variant_id": registration.variant_id, "planned_variants": 1}
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (json.dumps(evidence, sort_keys=True), run_id))
    return registration


def require_native_registration(conn, store, repository, spec, *, code_root,
                                as_of_month, scope="shadow", event_ids=None, expected=None):
    """Read-only admission; never create registration, ledger, or result rows."""
    registration, _ = _binding(repository, spec, code_root=code_root, as_of_month=as_of_month,
                               scope=scope, event_ids=event_ids, expected=expected)
    if not _existing(conn, store, registration):
        raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    return registration


def verify_native_registration(conn, store, registration):
    """Verify stored admission without claiming current-request revalidation."""
    if not isinstance(registration, NativeRegistration) or not _existing(conn, store, registration):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native outcome requires an intact registration")


def read_native_registration(conn, store, spec):
    """Load the stable reservation before validating the current execution request."""
    key = content_hash({"experiment_id": spec.experiment_id, "arm": spec.primary_arm_id})
    run_id = "native_" + key.removeprefix("sha256:")
    row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    try:
        ref = from_document(ArtifactRef, json.loads(row[0])["native_registration"])
        registration = NativeRegistration(run_id, store.read_verified(ref))
        verify_native_registration(conn, store, registration)
        if registration.document["spec_hash"] != spec.spec_hash:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "requested specification differs from registration")
        return registration
    except (ArtifactError, ValueError, KeyError, TypeError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration evidence is invalid") from None
