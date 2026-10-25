"""Exact native preregistration over the existing experiment catalog records."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    safe_relative_path,
    to_document,
)
from engine.v2.foundation.final_holdout import declaration_from_document, request_from_registration
from engine.v2.ops.catalog import decode_bounded_text, sql_text_budget, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import (
    NATIVE_GATE_ENTRYPOINT,
    NATIVE_GATE_RUNNERS,
    NATIVE_GATE_WORKER,
    ExperimentSpec,
    register_hypothesis_in_transaction,
    resolve_experiment_plan,
)
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.ops.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    magnitude_execution_document,
    magnitude_fitted_execution_document,
    magnitude_history_execution_document,
    magnitude_history_request,
    native_evidence_json,
    validate_magnitude_fitted_execution,
    validate_magnitude_history_execution,
    verified_registration_bytes,
)
from engine.v2.research._pricing import STRUCTURES
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_folds import magnitude_window, plan_magnitude_folds
from experiments.magnitude_history_inputs import history_binding

MAX_GATE_SWEEP_BYTES = 16 * 1024 * 1024


def read_native_gate_sweep(conn, experiment_id):
    """Read bounded stored text atomically; None means no seal, never invalid text."""
    encoding, limit = sql_text_budget(conn, MAX_GATE_SWEEP_BYTES)
    row = conn.execute("SELECT CASE WHEN typeof(document_json)='text' "
                       "AND length(CAST(document_json AS BLOB))<=? THEN CAST(document_json AS BLOB) END "
                       "FROM native_gate_sweeps WHERE experiment_id=?",
                       (limit, experiment_id)).fetchone()
    if row is None:
        return None
    try:
        return decode_bounded_text(row[0], encoding, MAX_GATE_SWEEP_BYTES)
    except (ValueError, UnicodeError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored sweep requires bounded text") from None


def decode_native_gate_sweep(payload, *, failure_code="EXPERIMENT_IDENTITY_CONFLICT"):
    """Bound UTF-8 JSON before decoding; callers choose stored R6 or new-seal R1."""
    try:
        if (not isinstance(payload, str) or len(payload) > MAX_GATE_SWEEP_BYTES
                or len(payload.encode("utf-8")) > MAX_GATE_SWEEP_BYTES):
            raise ValueError
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError
        return document
    except (ValueError, TypeError, RecursionError):
        raise fail(failure_code, "native sweep requires bounded JSON object evidence") from None


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


def prepare_native_registration(repository, spec, *, code_root, as_of_month, scope, event_ids,
             final_read=None, expected=None, magnitude_training_start=None, magnitude_labels_before=None,
             admission=None, magnitude_feature_snapshot_id=None, magnitude_history_start=None, magnitude_fit_model=None,
             snapshot=None, extra_environment=None):
    """Prepare immutable registration/spec inputs without reserving or evaluating."""
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "native registration requires an experiment specification")
    plan = resolve_experiment_plan(spec)
    if plan.runner == NATIVE_GATE_WORKER and "strategy" not in plan.economic_params:
        raise fail("INVALID_EXPERIMENT_SPEC", "native worker requires a registered strategy")
    if (plan.runner in NATIVE_GATE_RUNNERS
            and "strategy" in plan.economic_params and plan.economic_params["strategy"] not in STRUCTURES):
        raise fail("INVALID_EXPERIMENT_SPEC", "declared strategy is not a supported native structure")
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
    history_request = None
    if magnitude_feature_snapshot_id is not None or magnitude_history_start is not None:
        if execution is None or plan.runner != "experiments/v2_prediction.py":
            raise fail("INVALID_EXPERIMENT_SPEC", "stored history requires magnitude execution and its native runner")
        history_request = magnitude_history_request(magnitude_feature_snapshot_id, magnitude_history_start)
        execution = magnitude_history_execution_document(magnitude_labels_before)
    if magnitude_fit_model is not None:
        if history_request is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "fitted magnitude requires stored history")
        execution = magnitude_fitted_execution_document(magnitude_labels_before, magnitude_fit_model)
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
        entries = [plan.runner] + ([NATIVE_GATE_ENTRYPOINT] if plan.runner == NATIVE_GATE_WORKER else [])
        sources = source_closure(root, entries)
        if any((root / name).resolve() != root / name for name in sources):
            raise ValueError
    except (ArtifactError, OpsError, OSError, SyntaxError, ValueError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "native source closure is missing, malformed or indirect") from None
    try:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError
        if snapshot is None:
            snapshot = repository.resolve_pinned(scope)
        elif not isinstance(snapshot, SnapshotRef) or repository.resolve(snapshot.snapshot_id) != snapshot:
            raise ValueError
    except (DataError, ValueError, TypeError, KeyError):
        raise fail("SNAPSHOT_UNRESOLVED", "native registration requires a current committed snapshot") from None
    document = {
        "schema_version": "native_experiment_registration.v1.0",
        "spec_hash": spec.spec_hash, "execution_plan": plan.as_document(),
        "snapshot": to_document(snapshot), "scope": scope,
        "source_closure": sources, "environment": {**environment_identity(1), **(extra_environment or {})},
    }
    if declaration is not None:
        document.update(schema_version="native_experiment_registration.v1.1",
                        final_read=to_document(declaration))
    if magnitude is not None:
        document.update(schema_version="native_experiment_registration.v1.2", magnitude_request=magnitude)
    if execution is not None:
        document.update(schema_version="native_experiment_registration.v1.3", magnitude_execution=execution)
    if history_request is not None:
        document.update(schema_version="native_experiment_registration.v1.4",
                        magnitude_history_request=history_request)
    if magnitude_fit_model is not None:
        document["schema_version"] = "native_experiment_registration.v1.5"
    key = content_hash({"experiment_id": plan.experiment_id, "arm": plan.arms[0]})
    run_id = "native_" + key.removeprefix("sha256:")
    if admission is not None:
        original = verified_registration_bytes(*admission, run_id)
        if original is None and magnitude is not None:
            raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
        if original is not None:
            stored = NativeRegistration(run_id, original)
            if magnitude is not None or stored.document["schema_version"] in (
                    "native_experiment_registration.v1.2", "native_experiment_registration.v1.3",
                    "native_experiment_registration.v1.4", "native_experiment_registration.v1.5"):
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
        if history_request is not None:
            document["magnitude_history"] = history_binding(repository, snapshot, temporal, history_request,
                as_of_month=as_of_month, event_ids=event_ids)
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
    if magnitude_fit_model is not None:
        validate_magnitude_fitted_execution(document)
    elif history_request is not None:
        validate_magnitude_history_execution(document)
    return NativeRegistration(run_id, canonical_json(document).encode("utf-8")), spec


def native_registration_exists(conn, store, registration, *, max_evidence_bytes=None):
    """Read exact expected evidence: absent is false; changed or corrupt is R6."""
    try:
        ref = artifact_reference(registration.binding_json, registration.document["schema_version"])
        binding = verified_registration_bytes(conn, store, registration.run_id, expected_ref=ref,
                                              max_evidence_bytes=max_evidence_bytes)
        return binding is not None
    except (ArtifactError, ValueError, TypeError, KeyError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration differs from its immutable binding") from None


def register_native(conn, store, repository, spec, *, code_root, as_of_month,
                    scope="shadow", event_ids=None, final_read=None, magnitude_training_start=None,
                    magnitude_labels_before=None, magnitude_feature_snapshot_id=None, magnitude_history_start=None,
                    magnitude_fit_model=None, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES):
    """Explicitly reserve an exact current binding before any outcome is read.

    Publication precedes the short catalog transaction. A failed commit can
    leave an unreferenced complete object, never a partially admitted binding.
    """
    if not valid_registration_read_budget(max_evidence_bytes):
        raise fail("INVALID_EXPERIMENT_SPEC", "registration read requires a positive SQLite-safe byte budget")
    registration, spec = prepare_native_registration(repository, spec, code_root=code_root, as_of_month=as_of_month,
                                  scope=scope, event_ids=event_ids, final_read=final_read,
                                  magnitude_training_start=magnitude_training_start,
                                  magnitude_labels_before=magnitude_labels_before,
                                  magnitude_feature_snapshot_id=magnitude_feature_snapshot_id,
                                  magnitude_history_start=magnitude_history_start, magnitude_fit_model=magnitude_fit_model)
    return reserve_native_registration(conn, store, registration, spec, max_evidence_bytes=max_evidence_bytes)


def _require_sweep_binding(conn, registration, spec):
    """Constrain reservations after sealing; earlier non-roster arms stay independent."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_gate_sweeps'").fetchone() is None:
        return
    payload = read_native_gate_sweep(conn, spec.experiment_id)
    if payload is None:
        return
    try:
        matches = [item for item in decode_native_gate_sweep(payload)["registrations"] if item["run_id"] == registration.run_id]
        if canonical_json(matches) != canonical_json([{"run_id": registration.run_id, "binding": registration.document}]):
            raise ValueError
    except (ValueError, TypeError, KeyError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native reservation differs from sealed sweep") from None


def reserve_native_registration(conn, store, registration, spec, *, max_evidence_bytes=None):
    """Reserve prepared bytes; an optional bound covers new and stored evidence."""
    if max_evidence_bytes is None:
        _require_sweep_binding(conn, registration, spec)
    else:
        if not valid_registration_read_budget(max_evidence_bytes):
            raise fail("INVALID_EXPERIMENT_SPEC", "registration read requires a positive SQLite-safe byte budget")
        if len(registration.binding_json) > max_evidence_bytes:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration exceeds its UTF-8 byte budget",
                       details={"registration_bytes": len(registration.binding_json),
                                "max_registration_bytes": max_evidence_bytes})
    ref = artifact_reference(registration.binding_json, registration.document["schema_version"])
    evidence = {"native_registration": to_document(ref),
                "variant_id": registration.variant_id, "planned_variants": 1}
    payload = (json.dumps(evidence, sort_keys=True) if max_evidence_bytes is None else
               native_evidence_json(evidence, max_bytes=max_evidence_bytes))
    if max_evidence_bytes is not None:
        _require_sweep_binding(conn, registration, spec)
    if native_registration_exists(conn, store, registration, max_evidence_bytes=max_evidence_bytes):
        return registration
    store.publish_bytes(registration.binding_json, schema_ref=ref.schema_ref)
    with transaction(conn):
        _require_sweep_binding(conn, registration, spec)
        if native_registration_exists(conn, store, registration, max_evidence_bytes=max_evidence_bytes):
            return registration
        if conn.execute("SELECT 1 FROM hypotheses WHERE spec_hash=?", (spec.spec_hash,)).fetchone():
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native specification is already reserved")
        run_id, created = register_hypothesis_in_transaction(
            conn, spec, registration.variant_id, mode="primary", run_id=registration.run_id)
        if not created or run_id != registration.run_id:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration key is already reserved")
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (payload, run_id))
    return registration


def require_native_registration(conn, store, repository, spec, *, code_root,
                                as_of_month, scope="shadow", event_ids=None, final_read=None, expected=None,
                                magnitude_training_start=None, magnitude_labels_before=None,
                                magnitude_feature_snapshot_id=None, magnitude_history_start=None, magnitude_fit_model=None):
    """Read-only admission; never create registration, ledger, or result rows."""
    registration, _ = prepare_native_registration(repository, spec, code_root=code_root, as_of_month=as_of_month,
                               scope=scope, event_ids=event_ids, final_read=final_read, expected=expected,
                               magnitude_training_start=magnitude_training_start,
                               magnitude_labels_before=magnitude_labels_before, admission=(conn, store),
                               magnitude_feature_snapshot_id=magnitude_feature_snapshot_id,
                               magnitude_history_start=magnitude_history_start, magnitude_fit_model=magnitude_fit_model)
    if not native_registration_exists(conn, store, registration):
        raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    return registration


def verify_native_registration(conn, store, registration, *, max_evidence_bytes=None):
    """Verify stored admission without claiming current-request revalidation."""
    if not isinstance(registration, NativeRegistration) or not native_registration_exists(
            conn, store, registration, max_evidence_bytes=max_evidence_bytes):
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


def valid_registration_read_budget(value):
    """Share strict integer admission with CLI parsers without I/O or coercion."""
    return type(value) is int and 0 < value < 2**62


def read_native_registration(conn, store, spec, *, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES):
    """Load a bounded stable reservation before current-request validation."""
    if not valid_registration_read_budget(max_evidence_bytes):
        raise fail("INVALID_EXPERIMENT_SPEC", "registration read requires a positive SQLite-safe byte budget")
    key = content_hash({"experiment_id": spec.experiment_id, "arm": spec.primary_arm_id})
    run_id = "native_" + key.removeprefix("sha256:")
    binding = verified_registration_bytes(conn, store, run_id, max_evidence_bytes=max_evidence_bytes)
    if binding is None:
        raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    try:
        registration = NativeRegistration(run_id, binding)
        if registration.document["spec_hash"] != spec.spec_hash:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "requested specification differs from registration")
        return registration
    except (ArtifactError, ValueError, KeyError, TypeError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration evidence is invalid") from None
