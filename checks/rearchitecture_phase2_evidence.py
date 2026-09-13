#!/usr/bin/env python3
"""Strict validator for the Phase 2 gate's private evidence document.

Phase-2 guide §12.1 quotes the shape::

    PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

    Phase2Evidence:
      schema_version, code_hash, environment_hash
      snapshot_ref, legacy_snapshot_object_ref
      table_contract_mapping_hash
      import_receipt_refs, fault_matrix_ref
      dependency_plan_refs, comparison_receipt_ref, render_comparison_receipt_ref,
      rollback_receipt_ref
      expected_population, supported_population, compared_population
      authority_mode: shadow

``comparison_receipt_ref`` and ``render_comparison_receipt_ref`` are two
DIFFERENT receipts, not one shared field: D15 is legacy-vs-adapter SCORE
parity and D19 is v2-vs-legacy RENDER BUNDLE parity, over different
populations and different stage graphs. One field would let D15's receipt
silently stand in for D19's.

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

``validate_evidence`` returns ``(findings, field_ok, document_ok)``.
``document_ok`` covers only whole-document identity (schema/code/environment
hash, authority_mode) -- a failure there means NOTHING in the manifest can be
trusted. ``field_ok`` is per declared field (a ref resolves, a receipt's
verdict agrees, the populations are in order): a problem with ONE field (say
``fault_matrix_ref``) must not silently invalidate a DIFFERENT field's row
(D15's populations, D19's render receipt). The gate combines
``document_ok`` and the specific fields a row's ``evidence_fields`` name to
decide that row, never the other rows' fields.
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
    "render_comparison_receipt_ref",
    "rollback_receipt_ref",
)
LIST_REF_FIELDS = ("import_receipt_refs", "dependency_plan_refs")
#: Refs whose bytes must additionally decode to a comparison receipt with
#: verdict "agree" -- D15's score-parity receipt and D19's render-bundle-
#: parity receipt, checked independently of each other.
VERDICT_REF_FIELDS = ("comparison_receipt_ref", "render_comparison_receipt_ref")
POPULATION_FIELDS = ("expected_population", "supported_population", "compared_population")
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


def _check_refs(evidence: dict, artifact_root: Path, findings: list,
                field_ok: dict[str, bool]) -> dict[str, bytes]:
    """Resolve every declared ref; return field -> verified bytes for scalar refs."""
    resolved: dict[str, bytes] = {}
    for field in REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        data = _resolve(evidence[field], artifact_root, findings, field)
        field_ok[field] = data is not None
        if data is not None:
            resolved[field] = data
    for field in LIST_REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        items = evidence[field]
        if isinstance(items, bool) or not isinstance(items, list):
            findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
            field_ok[field] = False
            continue
        field_ok[field] = bool(items) and all(
            _resolve(item, artifact_root, findings, f"{field}[{i}]") is not None
            for i, item in enumerate(items))
    return resolved


def _check_populations(evidence: dict, findings: list, field_ok: dict[str, bool]) -> None:
    values = {}
    for field in POPULATION_FIELDS:
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            findings.append({"code": "POPULATION_EMPTY", "field": field})
            for f in POPULATION_FIELDS:
                field_ok[f] = False
            return
        values[field] = value
    ordered = (values["compared_population"] <= values["supported_population"]
              <= values["expected_population"])
    if not ordered:
        findings.append({"code": "POPULATION_COLLAPSED"})
    for f in POPULATION_FIELDS:
        field_ok[f] = ordered


def _check_verdicts(resolved: dict[str, bytes], findings: list,
                    field_ok: dict[str, bool]) -> None:
    for field in VERDICT_REF_FIELDS:
        if field not in resolved:
            continue
        try:
            payload = json.loads(resolved[field])
            agrees = isinstance(payload, dict) and payload.get("verdict") == "agree"
        except ValueError:
            agrees = False
        if not agrees:
            findings.append({"code": "VERDICT_NOT_AGREE", "field": field})
            field_ok[field] = False


def validate_evidence(evidence: dict, *, artifact_root: Path,
                      code_hash: str, environment_hash: str
                      ) -> tuple[list[dict], dict[str, bool], bool]:
    """Every finding the strict ``phase2_evidence.v1.0`` document can produce.

    Returns ``(findings, field_ok, document_ok)`` -- see the module docstring
    for what each covers. ``code_hash``/``environment_hash`` are the CURRENT
    tree's values, computed the same way the gate computes them; a mismatch
    means the evidence was produced against a different working tree or
    interpreter/library set.
    """
    findings: list[dict] = []
    if not isinstance(evidence, dict) or evidence.get("schema_version") != PHASE2_EVIDENCE_V1:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "schema_version"})
        return findings, {}, False
    document_ok = True
    if evidence.get("code_hash") != code_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "code_hash"})
        document_ok = False
    if evidence.get("environment_hash") != environment_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "environment_hash"})
        document_ok = False
    if evidence.get("authority_mode") != AUTHORITY_MODE:
        findings.append({"code": "AUTHORITY_NOT_SHADOW"})
        document_ok = False
    field_ok: dict[str, bool] = {}
    resolved = _check_refs(evidence, artifact_root, findings, field_ok)
    _check_populations(evidence, findings, field_ok)
    _check_verdicts(resolved, findings, field_ok)
    return findings, field_ok, document_ok
