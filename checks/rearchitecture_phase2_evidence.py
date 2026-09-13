#!/usr/bin/env python3
"""Strict validator for the Phase 2 gate's private evidence document.

Phase-2 guide §12.1 quotes the shape::

    PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

    Phase2Evidence:
      schema_version, code_hash, environment_hash
      snapshot_ref, legacy_snapshot_object_ref
      table_contract_mapping_hash
      import_receipt_refs, fault_matrix_ref
      dependency_plan_refs, comparison_receipt_ref, rollback_receipt_ref
      expected_population, supported_population, compared_population
      authority_mode: shadow

"The evidence validator verifies every referenced artifact, requires all
D01-D20 rows, rejects a code/environment mismatch, rejects zero or collapsed
populations, and requires the comparison verdict agree. It does not accept a
summary boolean in place of the referenced receipts."

This module never produces real evidence: it only checks a document someone
else hands it. Reference shape (documented here because ``ArtifactStore``'s
content-addressed ``objects/<xx>/<hash>`` layout does not fit a manifest whose
refs point at diverse artifact kinds a coordinator did not itself publish)::

    {"path": "<relative path under --artifact-root>", "content_hash": "sha256:<hex>"}

A list-valued field (``import_receipt_refs``, ``dependency_plan_refs``) is a
list of these. A bare boolean anywhere a ref belongs is refused outright: it
is exactly the "summary boolean in place of the referenced receipts" the
guide names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

#: Single-artifact reference fields (excludes ``table_contract_mapping_hash``,
#: which is a bare hash of reviewed content, not a pointer to stored bytes).
REF_FIELDS = (
    "snapshot_ref",
    "legacy_snapshot_object_ref",
    "fault_matrix_ref",
    "comparison_receipt_ref",
    "rollback_receipt_ref",
)
LIST_REF_FIELDS = ("import_receipt_refs", "dependency_plan_refs")
POPULATION_FIELDS = ("expected_population", "supported_population", "compared_population")
#: Checked by the gate (not this module): a manifest with the wrong
#: authority_mode is not "invalid" in the code/environment/population/verdict
#: sense this validator covers, so it carries no code of its own here. The
#: gate treats it as evidence that does not prove any tier-2 row.
AUTHORITY_MODE = "shadow"


def _resolve(ref: Any, artifact_root: Path, findings: list, field: str) -> bytes | None:
    if isinstance(ref, bool):
        findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
        return None
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) \
            or not isinstance(ref.get("content_hash"), str):
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    path = (artifact_root / ref["path"]).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError:
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    if not path.is_file():
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    data = path.read_bytes()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != ref["content_hash"]:
        findings.append({"code": "ARTIFACT_HASH_MISMATCH", "field": field})
        return None
    return data


def _check_refs(evidence: dict, artifact_root: Path, findings: list) -> dict[str, bytes]:
    """Resolve every declared ref; return field -> verified bytes for scalar refs."""
    resolved: dict[str, bytes] = {}
    for field in REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        data = _resolve(evidence[field], artifact_root, findings, field)
        if data is not None:
            resolved[field] = data
    for field in LIST_REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        items = evidence[field]
        if isinstance(items, bool) or not isinstance(items, list):
            findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
            continue
        for index, item in enumerate(items):
            _resolve(item, artifact_root, findings, f"{field}[{index}]")
    return resolved


def _check_populations(evidence: dict, findings: list) -> None:
    values = {}
    for field in POPULATION_FIELDS:
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            findings.append({"code": "POPULATION_EMPTY", "field": field})
            return
        values[field] = value
    if not (values["compared_population"] <= values["supported_population"]
            <= values["expected_population"]):
        findings.append({"code": "POPULATION_COLLAPSED"})


def _check_verdict(comparison_bytes: bytes | None, findings: list) -> None:
    if comparison_bytes is None:
        return
    try:
        payload = json.loads(comparison_bytes)
    except ValueError:
        findings.append({"code": "VERDICT_NOT_AGREE"})
        return
    if not isinstance(payload, dict) or payload.get("verdict") != "agree":
        findings.append({"code": "VERDICT_NOT_AGREE"})


def validate_evidence(evidence: dict, *, artifact_root: Path,
                      code_hash: str, environment_hash: str) -> list[dict]:
    """Every finding the strict ``phase2_evidence.v1.0`` document can produce.

    ``code_hash``/``environment_hash`` are the CURRENT tree's values, computed
    the same way the gate computes them; a mismatch means the evidence was
    produced against a different working tree or interpreter/library set.
    """
    findings: list[dict] = []
    if not isinstance(evidence, dict) or evidence.get("schema_version") != PHASE2_EVIDENCE_V1:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "schema_version"})
        return findings
    if evidence.get("code_hash") != code_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "code_hash"})
    if evidence.get("environment_hash") != environment_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "environment_hash"})
    resolved = _check_refs(evidence, artifact_root, findings)
    _check_populations(evidence, findings)
    _check_verdict(resolved.get("comparison_receipt_ref"), findings)
    return findings


def authority_mode_ok(evidence: dict) -> bool:
    return isinstance(evidence, dict) and evidence.get("authority_mode") == AUTHORITY_MODE
