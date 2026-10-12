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
from engine.v2.foundation.final_holdout import declaration_from_document, request_from_registration
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import (
    ExperimentSpec,
    register_hypothesis_in_transaction,
    resolve_experiment_plan,
)
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.ops.native_registration import (
    magnitude_execution_document,
    verified_registration_bytes,
)
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_folds import magnitude_window, plan_magnitude_folds


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


def _binding(repository, spec, *, code_root, as_of_month, scope, event_ids,
             final_read=None, expected=None, magnitude_training_start=None, magnitude_labels_before=None,
             admission=None):
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "native registration requires an experiment specification")
    plan = resolve_experiment_plan(spec)
    magnitude = None
    if magnitude_training_start is not None:
        if final_read is not None or plan.price_source != "computed_moves":
            raise fail("INVALID_EXPERIMENT_SPEC", "magnitude registration requires computed_moves and no final read")
        try:
            magnitude = magnitude_window(plan.folds, magnitude_training_start)
        except DataError:
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid magnitude temporal window") from None
    execution = None
    if magnitude_labels_before is not None:
        if magnitude is None or type(plan.seed) is not int or plan.seed != 0 or plan.economic_params:
            raise fail("INVALID_EXPERIMENT_SPEC", "magnitude baseline requires its window, seed zero and no economics")
        execution = magnitude_execution_document(magnitude_labels_before)
    try:
        declaration = None if final_read is None else declaration_from_document(to_document(final_read))
    except (ValueError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid final-read preregistration metadata") from None
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
    if declaration is not None:
        document.update(schema_version="native_experiment_registration.v1.1",
                        final_read=to_document(declaration))
    if magnitude is not None:
        document.update(schema_version="native_experiment_registration.v1.2", magnitude_request=magnitude)
    if execution is not None:
        document.update(schema_version="native_experiment_registration.v1.3", magnitude_execution=execution)
    key = content_hash({"experiment_id": plan.experiment_id, "arm": plan.arms[0]})
    run_id = "native_" + key.removeprefix("sha256:")
    if admission is not None:
        original = verified_registration_bytes(*admission, run_id)
        if original is None and magnitude is not None:
            raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
        if original is not None:
            stored = NativeRegistration(run_id, original)
            if magnitude is not None or stored.document["schema_version"] in (
                    "native_experiment_registration.v1.2", "native_experiment_registration.v1.3"):
                if expected is not None and expected != stored:
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "expected registration differs from original")
                expected = stored
    if expected is not None and (not document.keys() <= expected.document.keys()
            or any(expected.document[name] != value for name, value in document.items())):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native execution inputs changed before population admission")
    if magnitude is not None:
        if expected is not None and expected.document["holdouts"]["holdout_as_of_month"] != as_of_month:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "magnitude holdout month differs from registration")
        temporal = plan_magnitude_folds(repository, snapshot, as_of_month=as_of_month,
                                        event_ids=event_ids, **magnitude)
        evaluation = {key for fold in temporal["folds"] for key in fold["evaluation_event_ids"]}
        if execution is not None and any(event["event_date"] >= execution["evaluation_labels_before"]
                for event in temporal["events"] if event["event_id"] in evaluation):
            raise fail("INVALID_EXPERIMENT_SPEC", "magnitude evaluation events must precede the label cutoff")
        document["magnitude_plan"] = temporal
        selected_ids, context = [event["event_id"] for event in temporal["events"]], temporal
    else:
        events = load_population(repository, snapshot, as_of_month=as_of_month,
                                 purpose="selection", event_ids=event_ids)
        selected_ids, context = list(events["event_id"]), events.iloc[0]
    document.update({
        "event_ids": sorted(selected_ids),
        "holdouts": {name: context[name] for name in (
            "holdout_as_of_month", "random_membership_version", "rolling_membership_version")},
    })
    return NativeRegistration(run_id, canonical_json(document).encode("utf-8")), spec


def _existing(conn, store, registration):
    try:
        ref = artifact_reference(registration.binding_json, registration.document["schema_version"])
        binding = verified_registration_bytes(conn, store, registration.run_id, expected_ref=ref)
        return binding is not None
    except (ArtifactError, ValueError, TypeError, KeyError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding") from None


def register_native(conn, store, repository, spec, *, code_root, as_of_month,
                    scope="shadow", event_ids=None, final_read=None, magnitude_training_start=None,
                    magnitude_labels_before=None):
    """Explicitly reserve an exact current binding before any outcome is read.

    Publication precedes the short catalog transaction. A failed commit can
    leave an unreferenced complete object, never a partially admitted binding.
    """
    registration, spec = _binding(repository, spec, code_root=code_root, as_of_month=as_of_month,
                                  scope=scope, event_ids=event_ids, final_read=final_read,
                                  magnitude_training_start=magnitude_training_start,
                                  magnitude_labels_before=magnitude_labels_before)
    if _existing(conn, store, registration):
        return registration
    ref = store.publish_bytes(registration.binding_json,
                              schema_ref=registration.document["schema_version"])
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
                                as_of_month, scope="shadow", event_ids=None, final_read=None, expected=None,
                                magnitude_training_start=None, magnitude_labels_before=None):
    """Read-only admission; never create registration, ledger, or result rows."""
    registration, _ = _binding(repository, spec, code_root=code_root, as_of_month=as_of_month,
                               scope=scope, event_ids=event_ids, final_read=final_read, expected=expected,
                               magnitude_training_start=magnitude_training_start,
                               magnitude_labels_before=magnitude_labels_before, admission=(conn, store))
    if not _existing(conn, store, registration):
        raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    return registration


def verify_native_registration(conn, store, registration):
    """Verify stored admission without claiming current-request revalidation."""
    if not isinstance(registration, NativeRegistration) or not _existing(conn, store, registration):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native outcome requires an intact registration")


def prepare_final_holdout_request(conn, store, registration):
    """Derive a request from preregistered metadata only; grant no authority."""
    verify_native_registration(conn, store, registration)
    try:
        document = registration.document
        if canonical_json(document).encode() != registration.binding_json:
            raise ValueError
        return request_from_registration(document, registration.run_id,
                                         artifact_reference(registration.binding_json,
                                                            document["schema_version"]))
    except (ValueError, TypeError, KeyError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid final-read registration metadata") from None


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
