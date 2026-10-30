"""SIZE preflight receipt: issued once by preflight, verified cheaply by register."""
from __future__ import annotations

import dataclasses
import json
from datetime import date

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    DocumentError,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import ExperimentSpec, resolve_experiment_plan
from experiments.magnitude_search_request_versions import compile_request
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    NativeRegistration,
    load_registration_document,
)
from experiments.size_search_binding import (
    SCHEMA as REGISTRATION_SCHEMA,
    validate_panel_request,
    validator_identity,
)

SCHEMA = "size_preflight_receipt.v1"
_RECEIPT_KEYS = frozenset({"schema_version", "bindings", "validator", "run_id", "registration"})


def input_bindings(repository, spec, request, panel_request, *, scope, training_start,
                   labels_before):
    """Register-side cheap inputs only: hashes and resolved snapshot documents, no reads."""
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "SIZE receipt requires an experiment specification")
    validate_panel_request(panel_request)
    for value in (training_start, labels_before):
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except (TypeError, ValueError):
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid SIZE cutoff date")
    if not isinstance(scope, str) or not scope.strip():
        raise fail("SNAPSHOT_UNRESOLVED",
                   "native registration requires a current committed snapshot")
    return {
        "spec_hash": spec.spec_hash,
        "request": content_hash(request),
        "size_request": content_hash(compile_request(request)["request"]),
        "panel_request": content_hash(panel_request),
        "scope": scope,
        "training_start": training_start,
        "labels_before": labels_before,
        "scope_snapshot": to_document(repository.resolve_pinned(scope)),
        "snapshots": {name: to_document(repository.resolve(panel_request[name]))
                      for name in ("feature_snapshot_id", "history_snapshot",
                                   "price_history_snapshot")},
        "limits": {"request": request["limits"], "panel": panel_request["limits"],
                   "metadata": panel_request["metadata_limits"]},
    }


def issue(store, bindings, registration):
    """Publish the registration payload and its deterministic, timestamp-free receipt."""
    document = registration.document
    if document["snapshot"] != bindings["scope_snapshot"]:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                   "scope snapshot moved during preflight; rerun preflight",
                   details={"reason": "receipt_input_mismatch"})
    payload_ref = store.publish_bytes(registration.binding_json,
                                      schema_ref=REGISTRATION_SCHEMA)
    receipt = {"schema_version": SCHEMA, "bindings": bindings,
               "validator": validator_identity(), "run_id": registration.run_id,
               "registration": to_document(payload_ref)}
    return store.publish_bytes(canonical_json(receipt).encode("utf-8"), schema_ref=SCHEMA)


def admit(store, ref, bindings, spec):
    """Verify the receipt against current cheap inputs; return registration and prepared spec."""
    if not isinstance(spec, ExperimentSpec) or not isinstance(ref, ArtifactRef):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "SIZE receipt admission requires a specification and an artifact reference")
    if spec.input_files:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "native registration requires a Python runner and no file inputs")
    if ref.byte_size > MAX_NATIVE_EVIDENCE_BYTES:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE preflight receipt exceeds its byte budget",
                   details={"reason": "receipt_corrupt"})
    try:
        data = store.read_verified(ref)
    except ArtifactError as error:
        if error.code == "MISSING":
            raise fail("INVALID_EXPERIMENT_SPEC", "SIZE preflight receipt is missing",
                       details={"reason": "receipt_missing"}) from None
        if error.code == "INTEGRITY_FAILED":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                       "SIZE preflight receipt failed verification",
                       details={"reason": "receipt_corrupt"}) from None
        raise
    try:
        receipt = json.loads(data)
        if (type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS
                or receipt["schema_version"] != SCHEMA):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE preflight receipt is malformed",
                   details={"reason": "receipt_corrupt"}) from None
    if receipt["bindings"] != bindings:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE preflight receipt inputs differ",
                   details={"reason": "receipt_input_mismatch"})
    if receipt["validator"] != validator_identity():
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE preflight receipt validator is stale",
                   details={"reason": "receipt_validator_stale"})
    plan = resolve_experiment_plan(spec)
    expected_run_id = ("native_" + content_hash({"experiment_id": plan.experiment_id,
                                                 "arm": plan.arms[0]}).removeprefix("sha256:"))
    if type(receipt["run_id"]) is not str or receipt["run_id"] != expected_run_id:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                   "SIZE preflight receipt run_id is not bound to its plan",
                   details={"reason": "receipt_corrupt"})
    try:
        payload_ref = from_document(ArtifactRef, receipt["registration"])
        if payload_ref.byte_size > MAX_NATIVE_EVIDENCE_BYTES:
            raise ValueError
        payload = store.read_verified(payload_ref)
        if len(payload) > MAX_NATIVE_EVIDENCE_BYTES:
            raise ValueError
        logical = load_registration_document(store, payload)
        registration = NativeRegistration(receipt["run_id"], payload, logical)
    except ArtifactError as error:
        if error.code in ("MISSING", "INTEGRITY_FAILED"):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                       "SIZE preflight receipt payload failed verification",
                       details={"reason": "receipt_corrupt"}) from None
        raise
    except (DocumentError, ValueError, KeyError, TypeError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE preflight receipt payload is invalid",
                   details={"reason": "receipt_corrupt"}) from None
    prepared = dataclasses.replace(spec, economic_params=plan.as_document()["economic_params"],
                                   input_files=())
    document = registration.document
    try:
        panel_request = document["size_panel_request"]
        snapshots = bindings["snapshots"]
        bound = (content_hash(panel_request) == bindings["panel_request"]
                 and content_hash(document["size_request"]) == bindings["size_request"]
                 and all(panel_request[name] == snapshots[name]["snapshot_id"]
                         for name in ("feature_snapshot_id", "history_snapshot",
                                      "price_history_snapshot"))
                 and document["scope"] == bindings["scope"]
                 and document["magnitude_request"]["training_start"] == bindings["training_start"]
                 and document["magnitude_execution"]["evaluation_labels_before"]
                 == bindings["labels_before"]
                 and document["execution_plan"] == plan.as_document())
    except (KeyError, TypeError):
        bound = False
    if (not bound
            or document["spec_hash"] != prepared.spec_hash
            or document["snapshot"] != bindings["scope_snapshot"]
            or document["schema_version"] != "native_experiment_registration.v1.7"):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                   "receipt registration differs from its prepared inputs",
                   details={"reason": "receipt_corrupt"})
    return registration, prepared
