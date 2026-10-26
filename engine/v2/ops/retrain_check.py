"""Release acceptance and pool-only retrain check job.

The acceptance evidence is produced by the existing release acceptance gate
and arrives as a job-bound input. This worker verifies that evidence is for the
exact candidate catalog, then permits only a pool-only candidate whose model
binding metadata and actual model object bytes match the incumbent.
"""
from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path
from typing import Any

from engine.v2.foundation import content_hash, safe_relative_path
from engine.v2.foundation.artifacts import ArtifactError, _open_beneath
from engine.v2.models import deployment
from engine.v2.models.release_layout import (
    catalog_path,
    deployment_root,
    read_catalog,
    sha256_bytes,
)
from engine.v2.models.release_objects import read_regular_file
from engine.v2.ops.errors import fail
from engine.v2.ops.submission import JobKind, RetryPolicy

ACCEPTANCE_SCHEMA = "phase5_acceptance.v1.0"
RECEIPT_SCHEMA = "retrain_check_receipt.v1.0"
ACCEPTANCE_FILE = "acceptance.json"
MAX_ACCEPTANCE_BYTES = 8 * 1024 * 1024
MAX_MODEL_OBJECT_BYTES = 512 * 1024 * 1024


@dataclasses.dataclass(frozen=True, kw_only=True)
class RetrainCheckParameters:
    expected_ids: tuple[str, ...]
    release_root: str
    candidate_release_id: str
    incumbent_release_id: str
    input_bindings: dict[str, str] | None = None


def retrain_check_parameter_problems(_job: Any, params: Any) -> list[str]:
    problems = []
    if tuple(getattr(params, "expected_ids", ())) != ("retrain_check",):
        problems.append("expected_ids must be exactly ('retrain_check',)")
    for field in ("release_root", "candidate_release_id", "incumbent_release_id"):
        value = getattr(params, field, None)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field} must be a nonblank string")
    if getattr(params, "candidate_release_id", None) == getattr(
            params, "incumbent_release_id", None):
        problems.append("candidate and incumbent release ids must differ")
    bindings = getattr(params, "input_bindings", None)
    if not isinstance(bindings, dict) or ACCEPTANCE_FILE not in bindings:
        problems.append("input_bindings must bind acceptance.json")
    return problems


def retrain_check_job_kind() -> JobKind:
    return JobKind(
        name="retrain_check", worker="retrain_check", parameters=RetrainCheckParameters,
        resource_classes=frozenset({"validation"}), effects=("staged",),
        retry=RetryPolicy("none", 1), checkpoint_contract=RECEIPT_SCHEMA,
        namespaces=frozenset({"shadow", "smoke"}),
        validate=retrain_check_parameter_problems,
    )


def _acceptance_evidence(root: Path, release_root: Path,
                         candidate_release_id: str) -> dict:
    data = read_regular_file(root / ACCEPTANCE_FILE, MAX_ACCEPTANCE_BYTES)
    if data is None:
        raise fail("VALIDATION_FAILED", "release acceptance evidence is missing",
                   details={"code": "POOL_ACCEPTANCE_MISSING"})
    try:
        evidence = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise fail("VALIDATION_FAILED", "release acceptance evidence is invalid",
                   details={"code": "POOL_ACCEPTANCE_INVALID"}) from exc
    if not isinstance(evidence, dict) or evidence.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise fail("VALIDATION_FAILED", "release acceptance evidence has the wrong schema",
                   details={"code": "POOL_ACCEPTANCE_INVALID"})
    catalog = read_catalog(catalog_path(release_root, candidate_release_id), candidate_release_id)
    if catalog is None:
        raise fail("VALIDATION_FAILED", "candidate release catalog is missing",
                   details={"code": "CANDIDATE_CATALOG_MISSING"})
    unhashed = {key: value for key, value in evidence.items()
                if key not in ("runtime_ms", "report", "evidence_hash")}
    if (evidence.get("evidence_hash") != content_hash(unhashed)
            or evidence.get("release_id") != candidate_release_id
            or evidence.get("manifest_hash") != catalog.get("manifest_hash")):
        raise fail("INPUT_CHANGED", "release acceptance evidence does not match candidate",
                   details={"code": "POOL_ACCEPTANCE_MISMATCH"})
    if (evidence.get("release_ok") is not True
            or evidence.get("status") not in ("PASS", "RELEASE_PASS")
            or evidence.get("findings") != []):
        raise fail("VALIDATION_FAILED", "candidate release acceptance failed",
                   details={"code": "POOL_ACCEPTANCE_FAILED"})
    return evidence


def _binding_metadata(release) -> dict[tuple[str, str], tuple]:
    rows = {}
    for binding in release.bindings:
        key = (binding.role, binding.strategy_id)
        if key in rows:
            raise fail("VALIDATION_FAILED", "release has duplicate model bindings",
                       details={"code": "MODEL_BINDING_INVALID"})
        rows[key] = (binding.binding_id, binding.model_id, binding.decision_clock_id,
                     binding.adapter, binding.feature_order, binding.output_names,
                     binding.schema_version)
    return rows


def _model_objects(release_root: Path, release) -> dict[tuple[str, str, str], bytes]:
    store = deployment_root(release_root)
    result = {}
    for binding in release.bindings:
        for member in binding.members:
            try:
                relative = safe_relative_path(member.path)
                fd = _open_beneath(store, relative)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise OSError("model object is not a regular file")
                    handle = os.fdopen(fd, "rb")
                    fd = -1
                    with handle:
                        payload = handle.read(MAX_MODEL_OBJECT_BYTES + 1)
                    if len(payload) > MAX_MODEL_OBJECT_BYTES:
                        raise ValueError("model object exceeds the read limit")
                finally:
                    if fd >= 0:
                        os.close(fd)
            except (ArtifactError, OSError, ValueError, TypeError) as exc:
                raise fail("VALIDATION_FAILED", "model object could not be verified",
                           details={"code": "MODEL_OBJECT_INVALID"}) from exc
            if sha256_bytes(payload) != member.content_hash:
                raise fail("VALIDATION_FAILED", "model object hash does not match its binding",
                           details={"code": "MODEL_OBJECT_HASH_MISMATCH"})
            key = (binding.role, binding.strategy_id, member.name)
            if key in result:
                raise fail("VALIDATION_FAILED", "release has duplicate model objects",
                           details={"code": "MODEL_OBJECT_INVALID"})
            result[key] = payload
    return result


def run_retrain_check_worker(parameters, root: Path) -> dict:
    values = parameters if isinstance(parameters, dict) else dataclasses.asdict(parameters)
    candidate_id = values["candidate_release_id"]
    incumbent_id = values["incumbent_release_id"]
    release_root = Path(values["release_root"])
    evidence = _acceptance_evidence(Path(root), release_root, candidate_id)
    candidate = deployment.read_manifest(deployment_root(release_root), candidate_id)
    incumbent = deployment.read_manifest(deployment_root(release_root), incumbent_id)
    if candidate is None or incumbent is None:
        raise fail("VALIDATION_FAILED", "candidate or incumbent release is not staged",
                   details={"code": "RELEASE_NOT_STAGED"})
    candidate_meta = _binding_metadata(candidate.release)
    incumbent_meta = _binding_metadata(incumbent.release)
    candidate_objects = _model_objects(release_root, candidate.release)
    incumbent_objects = _model_objects(release_root, incumbent.release)
    if candidate_meta != incumbent_meta or candidate_objects != incumbent_objects:
        raise fail("ACCEPTANCE_RULE_UNDEFINED",
                   "model-changing candidate has no decided non-regression pass rule",
                   details={"code": "ACCEPTANCE_RULE_UNDEFINED"})
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "passed",
        "result": "pool_only_noop", "candidate_release_id": candidate_id,
        "candidate_release_hash": candidate.release_hash,
        "incumbent_release_id": incumbent_id,
        "incumbent_release_hash": incumbent.release_hash,
        "check_candidate_release_hash": incumbent.release_hash,
        "acceptance_evidence_hash": evidence["evidence_hash"],
    }
    output = Path(root) / "retrain_check_receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return {"outputs": [{"name": "retrain_check_receipt", "path": output.name,
                          "schema": RECEIPT_SCHEMA}],
            "completed_ids": values["expected_ids"], "receipt": receipt}
