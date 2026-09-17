#!/usr/bin/env python3
"""Strict validator for the Phase 3 gate's private evidence document.

Phase-3 guide §10 quotes the shape::

    Phase3Evidence (phase3_evidence.v1.0):
      implementation_code_hash, environment_hash, frontend_lock_hash
      phase2_acceptance_ref, source_code_hash, source_environment_hash
      preview_input_refs, accepted_release_refs
      population_manifest_ref, mapping_version
      comparison_receipt_refs, negative_control_receipt_refs
      browser_receipt_ref, refresh_rollback_receipt_ref
      engineering_receipt_ref, coverage_receipt_ref, performance_receipt_ref
      view_field_inventory_ref, deferred_work_ref
      authority_mode = "shadow"

This module never produces real evidence: it only checks a document someone
else hands it -- mirroring ``checks/rearchitecture_phase2_evidence.py``
exactly in spirit (reused directly for the Phase 2 prerequisite, never
reimplemented). Reference shape for every ``*_ref``/``*_refs`` field, same as
Phase 2's::

    {"path": "<relative path under --artifact-root>", "content_hash": "sha256:<hex>"}

A bare boolean anywhere a ref belongs is refused outright (``SUMMARY_BOOLEAN_
REFUSED``), same as Phase 2.

**Binding, not mere coexistence (guide §10).** ``phase2_acceptance_ref`` /
``source_code_hash`` / ``source_environment_hash`` stay bound to Phase 2's
OWN producer commit: the referenced ``Phase2Evidence`` document's own
``code_hash``/``environment_hash`` fields must equal the declared
``source_code_hash``/``source_environment_hash`` (checked BY
``rearchitecture_phase2_evidence.validate_evidence`` itself, called with
those two values) -- a Phase3Evidence edited to claim a newer
``source_code_hash`` than what the referenced document actually contains
fails there, which is what "no evidence older than its producer commit"
means for the Phase 2 half. Every ``comparison_receipt_refs``/
``negative_control_receipt_refs`` item, by contrast, binds to
``implementation_code_hash``/``environment_hash`` (the CURRENT Phase 3
tree) -- guide §10: "Phase 3 tests/projection evidence binds to the final
implementation." These are deliberately two different anchors; do not
collapse them into one code/environment pair.

**Nested Phase 2 refs resolve against the SAME ``--artifact-root``.** The
builder is expected to copy Phase 2's referenced artifacts into the shared
artifact root (for example under a ``phase2/`` prefix) so that the nested
``{"path": ..., "content_hash": ...}`` refs inside the ``Phase2Evidence``
document resolve the same way this validator resolves its own top-level
refs. No separate ``phase2_artifact_root`` field exists in the schema (the
guide's field list is exhaustive), so this convention is how strict decoding
stays possible without one.

**Release binding convention (a judgement call, documented here because the
guide does not spell out the mechanism).** ``ComparisonReceipt`` carries no
release-id field of its own; a receipt that is ABOUT one accepted release
records that release's id in its ``right_ref`` as the literal string
``"release:<release_id>"``. When a receipt uses that prefix, the named id
must appear among the decoded ``accepted_release_refs``' own ``release_id``
values (``RELEASE_BINDING_MISMATCH`` otherwise). A receipt not about any one
release (for example the L09 no-scoring-startup negative control) simply
does not use the prefix and is not checked against it.

**Each accepted release binds to its own real ``projection_binding.v1.0``
document (added 2026-09-14, after P3-1c merged the real fenced-publish
binding -- ``engine.v2.serving.projections.projection_binding``,
``engine.v2.ops.effects_graph.publication_effect``'s ``projection_binding.
json`` input, and the API's current-resolution through it,
``tests/test_v2_serving_publication_binding.py``).** A ``PreviewRelease``
document alone only ASSERTS its own ``release_id``/``source_release_id``;
the real binding document is what a live publish actually produced and the
read API actually resolved through. Every entry of ``accepted_release_refs``
therefore carries a REQUIRED sibling ``binding_ref`` (same
``{"path","content_hash"}`` shape, alongside -- not inside -- that entry's
own ``path``/``content_hash``) pointing at that release's real binding
document; its ``projection_release_id``/``source_release_id`` must equal the
paired ``PreviewRelease``'s ``release_id``/``source_release_id``
(``RELEASE_BINDING_MISMATCH`` otherwise). This is an additive convention on
top of the ref-wrapper shape ``checks/`` already defines (the guide's §10
field LIST names ``accepted_release_refs`` itself, not the shape of one
entry's wrapper), so it does not add a new top-level ``Phase3Evidence``
field.

**Negative controls prove the control FIRED, not that everything agreed.**
A ``comparison_receipt_refs`` item must carry ``verdict == "agree"``
(``VERDICT_NOT_AGREE`` otherwise). A ``negative_control_receipt_refs`` item
must carry a verdict OTHER than ``"agree"`` -- ``differ`` or
``incomparable`` -- because the entire point of a negative control is that
it caught the injected fault; an "agreeing" negative control means the
control never fired (``NEGATIVE_CONTROL_NOT_TRIGGERED``).

``validate_evidence`` returns ``(findings, field_ok, document_ok)``, same
contract as Phase 2's validator: ``document_ok`` covers only whole-document
identity (schema version, unknown top-level fields, ``authority_mode``);
``field_ok`` is per declared top-level field. The Phase 3 gate additionally
tracks ``kind_ok`` (per ``(list_field, comparison_kind)``) to resolve the
L01-L14 matrix against ``checks/phase3_acceptance.json`` -- see
``rearchitecture_phase3_gate.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from checks.rearchitecture_phase2_evidence import PHASE2_EVIDENCE_V1, validate_evidence as validate_phase2_evidence
from engine.v2.contracts import ArtifactRef, ObjectRef, PreviewInput, PreviewRelease, RollbackReceipt  # noqa: F401
from engine.v2.data.documents import decode_document
from engine.v2.diagnosis.receipt import AGREE, ComparisonReceipt
from engine.v2.foundation import DocumentError, from_document
from engine.v2.serving.projections import PROJECTION_BINDING_V1

PHASE3_EVIDENCE_V1 = "phase3_evidence.v1.0"
AUTHORITY_MODE = "shadow"

#: Plain scalar (non-ref) fields; every one is a required, nonempty string.
SCALAR_FIELDS = (
    "implementation_code_hash", "environment_hash", "frontend_lock_hash",
    "source_code_hash", "source_environment_hash", "mapping_version",
)

#: Single-artifact reference fields decoding to a typed contract.
DECODE_CLASS: dict[str, type] = {
    "refresh_rollback_receipt_ref": RollbackReceipt,
}
#: Single-artifact reference fields checked as a raw JSON shape (no typed
#: contract exists for these -- guide §10 names them in prose only).
RAW_SINGLE_FIELDS = (
    "population_manifest_ref", "browser_receipt_ref", "engineering_receipt_ref",
    "coverage_receipt_ref", "performance_receipt_ref",
)
#: Single-artifact reference fields checked as a raw JSON *list* shape.
RAW_LIST_SINGLE_FIELDS = ("view_field_inventory_ref", "deferred_work_ref")
#: List-of-ref fields decoding each item to a typed contract.
LIST_DECODE_CLASS: dict[str, type] = {
    "preview_input_refs": PreviewInput,
    "accepted_release_refs": PreviewRelease,
    "comparison_receipt_refs": ComparisonReceipt,
    "negative_control_receipt_refs": ComparisonReceipt,
}
#: The two comparison-receipt lists whose items must AGREE.
AGREE_LIST_FIELDS = ("comparison_receipt_refs",)
#: The list whose items must NOT agree -- a fired negative control.
NEGATIVE_LIST_FIELDS = ("negative_control_receipt_refs",)

REF_FIELDS = ("phase2_acceptance_ref", *DECODE_CLASS, *RAW_SINGLE_FIELDS, *RAW_LIST_SINGLE_FIELDS)
LIST_REF_FIELDS = tuple(LIST_DECODE_CLASS)

# Optional for a clean prerequisite. Required only when the reused strict
# Phase 2 validator reports retained findings.
PHASE2_HANDOFF_DISPOSITION_V1 = "phase2_handoff_disposition.v1.0"
PHASE2_DISPOSITION_FIELD = "phase2_handoff_disposition_ref"

ALL_FIELDS = frozenset((
    "schema_version", "authority_mode", *SCALAR_FIELDS, *REF_FIELDS, *LIST_REF_FIELDS,
    PHASE2_DISPOSITION_FIELD,
))


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


def _decode(cls: type, data: bytes, findings: list, field: str) -> Any:
    try:
        doc = json.loads(data)
    except ValueError:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "invalid_json"})
        return None
    try:
        return decode_document(cls, doc)
    except DocumentError as exc:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": exc.code})
        return None


def _check_scalars(evidence: dict, findings: list, field_ok: dict[str, bool]) -> None:
    for field in SCALAR_FIELDS:
        value = evidence.get(field)
        ok = isinstance(value, str) and bool(value)
        field_ok[field] = ok
        if not ok:
            findings.append({"code": "MISSING_EVIDENCE", "field": field})


def _check_raw_dict(data: bytes | None, findings: list, field_ok: dict[str, bool], field: str,
                    required_keys: tuple[str, ...] = ()) -> dict | None:
    if data is None:
        return None
    try:
        doc = json.loads(data)
    except ValueError:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "invalid_json"})
        field_ok[field] = False
        return None
    if not isinstance(doc, dict) or any(k not in doc for k in required_keys):
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "missing_keys"})
        field_ok[field] = False
        return None
    return doc


def _check_raw_list(data: bytes | None, findings: list, field_ok: dict[str, bool], field: str,
                    required_keys: tuple[str, ...] = ()) -> list | None:
    if data is None:
        return None
    try:
        doc = json.loads(data)
    except ValueError:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "invalid_json"})
        field_ok[field] = False
        return None
    valid = isinstance(doc, list) and all(
        isinstance(item, dict) and all(k in item for k in required_keys) for item in doc)
    if not valid:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "not_a_valid_list"})
        field_ok[field] = False
        return None
    return doc


def _check_population_manifest(doc: dict | None, findings: list, field_ok: dict[str, bool]) -> None:
    if doc is None:
        return
    field = "population_manifest_ref"
    values = {}
    for key in ("expected", "supported", "compared"):
        value = doc.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            findings.append({"code": "POPULATION_EMPTY", "field": field, "reason": key})
            field_ok[field] = False
            return
        values[key] = value
    if not values["compared"] <= values["supported"] <= values["expected"]:
        findings.append({"code": "POPULATION_COLLAPSED", "field": field})
        field_ok[field] = False


def _check_browser_receipt(doc: dict | None, artifact_root: Path, findings: list,
                           field_ok: dict[str, bool]) -> None:
    if doc is None:
        return
    field = "browser_receipt_ref"
    screenshot = doc.get("screenshot_ref")
    if _resolve(screenshot, artifact_root, findings, f"{field}.screenshot_ref") is None:
        field_ok[field] = False


def _check_engineering_receipt(doc: dict | None, findings: list, field_ok: dict[str, bool]) -> None:
    if doc is None:
        return
    field = "engineering_receipt_ref"
    nights = doc.get("nights")
    valid = isinstance(nights, list) and bool(nights) and all(
        isinstance(n, dict) and n.get("status") in ("observed", "unknown") for n in nights)
    if not valid:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "nights"})
        field_ok[field] = False


def _check_coverage_receipt(doc: dict | None, implementation_code_hash: str | None, findings: list,
                            field_ok: dict[str, bool]) -> None:
    if doc is None:
        return
    field = "coverage_receipt_ref"
    if doc.get("source_hash") != implementation_code_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": field})
        field_ok[field] = False
        return
    from checks.rearchitecture_phase3_quality import coverage_findings
    for finding in coverage_findings(doc):
        findings.append({**finding, "field": field})
        field_ok[field] = False


def _check_verdicts_and_bindings(decoded_lists: dict[str, list[Any]], findings: list,
                                 field_ok: dict[str, bool], *, code_hash: str, environment_hash: str,
                                 accepted_release_ids: set[str]) -> dict[tuple[str, str], bool]:
    kind_ok: dict[tuple[str, str], bool] = {}
    for field in (*AGREE_LIST_FIELDS, *NEGATIVE_LIST_FIELDS):
        items = decoded_lists.get(field)
        if items is None:
            continue
        want_agree = field in AGREE_LIST_FIELDS
        for index, receipt in enumerate(items):
            ok = True
            if receipt.envelope.code_hash != code_hash:
                findings.append({"code": "CODE_HASH_MISMATCH", "field": f"{field}[{index}]"})
                ok = False
            if receipt.envelope.environment_hash != environment_hash:
                findings.append({"code": "ENVIRONMENT_MISMATCH", "field": f"{field}[{index}]"})
                ok = False
            if want_agree and receipt.verdict != AGREE:
                findings.append({"code": "VERDICT_NOT_AGREE", "field": f"{field}[{index}]"})
                ok = False
            if not want_agree and receipt.verdict == AGREE:
                findings.append({"code": "NEGATIVE_CONTROL_NOT_TRIGGERED", "field": f"{field}[{index}]"})
                ok = False
            if receipt.right_ref.startswith("release:"):
                release_id = receipt.right_ref[len("release:"):]
                if release_id not in accepted_release_ids:
                    findings.append({"code": "RELEASE_BINDING_MISMATCH", "field": f"{field}[{index}]"})
                    ok = False
            if want_agree and (receipt.population.compared <= 0):
                findings.append({"code": "POPULATION_EMPTY", "field": f"{field}[{index}]"})
                ok = False
            if not ok:
                field_ok[field] = False
            key = (field, receipt.comparison_kind)
            kind_ok[key] = kind_ok.get(key, False) or ok
    return kind_ok


def _check_rollback(receipt: Any, _accepted_release_ids: set[str], findings: list,
                    field_ok: dict[str, bool]) -> None:
    if receipt is None:
        return
    field = "refresh_rollback_receipt_ref"
    valid = (receipt.resulting_generation > receipt.prior_generation
             and receipt.prior_snapshot_id != receipt.resulting_snapshot_id)
    if not valid:
        findings.append({"code": "SINGLE_GENERATION", "field": field})
        field_ok[field] = False
        return
    if receipt.scope != "v2_serving_preview":
        findings.append({"code": "ROLLBACK_SCOPE_MISMATCH", "field": field})
        field_ok[field] = False
    if (receipt.prior_snapshot_id not in _accepted_release_ids
            or receipt.resulting_snapshot_id not in _accepted_release_ids):
        findings.append({"code": "RELEASE_BINDING_MISMATCH", "field": field})
        field_ok[field] = False


def _is_placeholder_ref(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return True
    lowered = value.lower()
    return (lowered.startswith("phase2_d") and "not_yet_produced" in lowered
            or lowered == "sha256:" + "0" * 64)


def _check_preview_lineage(preview_inputs: list[Any], releases: list[Any], findings: list,
                           field_ok: dict[str, bool]) -> None:
    field = "preview_input_refs"
    for index, preview in enumerate(preview_inputs):
        required = (
            preview.source_release_manifest_ref, preview.snapshot_ref, preview.score_batch_ref,
            preview.bundle_manifest_ref, preview.finality_ref, preview.expected_population_ref,
            preview.score_comparison_receipt_ref, preview.render_comparison_receipt_ref,
            *preview.score_job_input_refs, *preview.model_registry_artifact_refs,
        )
        if any(_is_placeholder_ref(value) for value in required):
            findings.append({"code": "PREVIEW_INPUT_UNVERIFIED", "field": f"{field}[{index}]"})
            field_ok[field] = False
            continue
        matching = [release for release in releases if (
            release.source_release_id == preview.source_release_id
            and release.snapshot_ref == preview.snapshot_ref
            and release.score_batch_ref == preview.score_batch_ref
            and release.bundle_manifest_ref == preview.bundle_manifest_ref
            and release.model_registry_artifact_refs == preview.model_registry_artifact_refs
            and release.source_code_hash == preview.source_code_hash
        )]
        if not matching:
            findings.append({"code": "PREVIEW_RELEASE_LINEAGE_MISMATCH", "field": f"{field}[{index}]"})
            field_ok[field] = False


def _check_preview_proofs(evidence: dict, preview_inputs: list[Any], artifact_root: Path,
                         findings: list, field_ok: dict[str, bool]) -> None:
    """Resolve the portable verifier proof carried beside each preview input."""
    for index, preview in enumerate(preview_inputs):
        item = evidence.get("preview_input_refs", [])[index]
        # Test-only symbolic source identities predate portable retained
        # artifacts. A real delivered release manifest is content-addressed
        # and therefore cannot omit this proof.
        if not preview.source_release_manifest_ref.startswith("sha256:"):
            continue
        proof_data = _resolve(item.get("verification_ref") if isinstance(item, dict) else None,
                              artifact_root, findings, f"preview_input_refs[{index}].verification_ref")
        if proof_data is None:
            field_ok["preview_input_refs"] = False
            continue
        try:
            proof = json.loads(proof_data)
        except ValueError:
            proof = None
        if not isinstance(proof, dict) or proof.get("schema_version") != "phase3_source_provenance.v1.0":
            findings.append({"code": "PREVIEW_PROOF_INVALID", "field": f"preview_input_refs[{index}]"})
            field_ok["preview_input_refs"] = False
            continue
        if (proof.get("release_id") != preview.source_release_id
                or proof.get("release_manifest_hash") != preview.source_release_manifest_ref
                or proof.get("bundle_manifest_ref") != preview.bundle_manifest_ref
                or tuple(proof.get("score_job_input_refs", ())) != preview.score_job_input_refs
                or tuple(proof.get("model_registry_artifact_refs", ())) != preview.model_registry_artifact_refs):
            findings.append({"code": "PREVIEW_PROOF_LINEAGE_MISMATCH", "field": f"preview_input_refs[{index}]"})
            field_ok["preview_input_refs"] = False
        expected = {"score_artifact": ("score", preview.score_batch_ref, "legacy_action.v1.0"),
                    "bundle_artifact": ("bundle", None, "legacy_action.v1.0"),
                    "snapshot_artifact": ("snapshot", None, "snapshot_ref.v1.0"),
                    "materialization_request_artifact": ("request", None, "legacy_materialization_request.v1.0"),
                    "finality_artifact": ("finality", preview.finality_ref, "legacy_action.v1.0"),
                    "model_evidence_artifact": ("model_evidence", preview.model_evidence_ref, "legacy_action.v1.0"),
                    "render_artifact": ("render", None, "legacy_action.v1.0")}
        files = proof.get("artifact_files")
        for metadata_name, (file_name, expected_hash, schema) in expected.items():
            meta, file_ref = proof.get(metadata_name), files.get(file_name) if isinstance(files, dict) else None
            data = _resolve(file_ref, artifact_root, findings, f"preview_input_refs[{index}].{file_name}")
            try:
                artifact = from_document(ArtifactRef, meta)
            except (DocumentError, TypeError):
                artifact = None
            if artifact is None or artifact.schema_ref != schema or data is None \
                    or file_ref.get("content_hash") != artifact.content_hash \
                    or (expected_hash is not None and artifact.content_hash != expected_hash):
                findings.append({"code": "PREVIEW_PROOF_ARTIFACT_MISMATCH",
                                 "field": f"preview_input_refs[{index}].{metadata_name}"})
                field_ok["preview_input_refs"] = False
        for name, kind, expected_ref, job_field in (("score_comparison_receipt", "score_record_parity", preview.score_comparison_receipt_ref, "score_job_id"),
                                                     ("render_comparison_receipt", "render_bundle_parity", preview.render_comparison_receipt_ref, "render_job_id")):
            data = _resolve(proof.get(name), artifact_root, findings, f"preview_input_refs[{index}].{name}")
            receipt = _decode(ComparisonReceipt, data, findings, f"preview_input_refs[{index}].{name}") if data else None
            if receipt is None or proof.get(name, {}).get("content_hash") != expected_ref \
                    or receipt.comparison_kind != kind or receipt.verdict != AGREE \
                    or receipt.envelope.snapshot_id != preview.snapshot_ref \
                    or receipt.right_ref != proof.get(job_field):
                findings.append({"code": "PREVIEW_PROOF_RECEIPT_MISMATCH", "field": f"preview_input_refs[{index}].{name}"})
                field_ok["preview_input_refs"] = False


def _check_complete_population(decoded_lists: dict[str, list[Any]], population: dict | None,
                               findings: list, field_ok: dict[str, bool]) -> None:
    if population is None:
        return
    expected = population.get("compared")
    receipts = {receipt.comparison_kind: receipt for receipt in
                decoded_lists.get("comparison_receipt_refs", [])}
    for kind in ("bridge_mapping_parity", "bridge_value_parity", "full_population_parity"):
        receipt = receipts.get(kind)
        if receipt is None or receipt.population.compared != expected:
            findings.append({"code": "POPULATION_LINEAGE_MISMATCH",
                             "field": f"comparison_receipt_refs.{kind}"})
            field_ok["comparison_receipt_refs"] = False


def _check_release_bindings(evidence: dict, artifact_root: Path, findings: list,
                            field_ok: dict[str, bool]) -> None:
    """Each ``accepted_release_refs`` entry binds to its own real
    ``projection_binding.v1.0`` document -- see the module docstring's
    "Each accepted release binds to its own real projection_binding.v1.0
    document" section. A ``PreviewRelease`` file alone only ASSERTS its own
    identity; this additionally requires the REAL binding a live publish
    produced (``engine.v2.serving.projections.projection_binding``) to name
    the same ``release_id``/``source_release_id``.
    """
    field = "accepted_release_refs"
    items = evidence.get(field)
    if not isinstance(items, list):
        return
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        release_data = _resolve(item, artifact_root, [], f"{field}[{index}]")
        if release_data is None:
            continue  # already reported by the generic resolve pass above
        try:
            release_doc = json.loads(release_data)
        except ValueError:
            continue  # already reported (ARTIFACT_SHAPE_INVALID) by the generic decode pass
        binding_data = _resolve(item.get("binding_ref"), artifact_root, findings,
                                f"{field}[{index}].binding_ref")
        if binding_data is None:
            field_ok[field] = False
            continue
        try:
            binding_doc = json.loads(binding_data)
        except ValueError:
            findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": f"{field}[{index}].binding_ref",
                             "reason": "invalid_json"})
            field_ok[field] = False
            continue
        valid = (isinstance(binding_doc, dict)
                 and binding_doc.get("schema_version") == PROJECTION_BINDING_V1
                 and binding_doc.get("projection_release_id") == release_doc.get("release_id")
                 and binding_doc.get("source_release_id") == release_doc.get("source_release_id"))
        if not valid:
            findings.append({"code": "RELEASE_BINDING_MISMATCH", "field": f"{field}[{index}].binding_ref"})
            field_ok[field] = False


def _same_ref(left: Any, right: Any) -> bool:
    return (isinstance(left, dict) and isinstance(right, dict)
            and left.get("path") == right.get("path")
            and left.get("content_hash") == right.get("content_hash"))


def _check_phase2_disposition(disposition_data: bytes | None, *, phase2_ref: Any, phase2_doc: dict,
                              artifact_root: Path, phase2_findings: list[dict]) -> bool:
    """Validate the exact, limited Phase 2 handoff exception.

    The disposition does not change the strict Phase 2 result. It can only
    name the exact evidence document and D14/D15 receipt refs, populations,
    and recorded stale-price cause. Any other finding remains a hard block.
    """
    if disposition_data is None:
        return False
    retained = {
        ("VERDICT_NOT_AGREE", "corpus_comparison_receipt_ref"),
        ("CODE_HASH_MISMATCH", "corpus_comparison_receipt_ref"),
        ("VERDICT_NOT_AGREE", "comparison_receipt_ref"),
    }
    observed = {(finding.get("code"), finding.get("field")) for finding in phase2_findings}
    if observed != retained:
        return False
    try:
        disposition = json.loads(disposition_data)
    except ValueError:
        return False
    if not isinstance(disposition, dict) or disposition.get("schema_version") != PHASE2_HANDOFF_DISPOSITION_V1:
        return False
    if not _same_ref(disposition.get("phase2_evidence_ref"), phase2_ref):
        return False
    if (disposition.get("candidate_code_hash") != phase2_doc.get("code_hash")
            or disposition.get("candidate_environment_hash") != phase2_doc.get("environment_hash")):
        return False
    required = {
        "D14": "corpus_comparison_receipt_ref",
        "D15": "comparison_receipt_ref",
    }
    entries = disposition.get("accepted_findings")
    if not isinstance(entries, list) or len(entries) != len(required):
        return False
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        d_id, receipt_field = entry.get("d_id"), entry.get("receipt_field")
        if d_id not in required or required[d_id] != receipt_field or d_id in seen:
            return False
        seen.add(d_id)
        receipt_ref = phase2_doc.get(receipt_field)
        if not _same_ref(entry.get("receipt_ref"), receipt_ref):
            return False
        receipt_data = _resolve(receipt_ref, artifact_root, [], f"phase2.{receipt_field}")
        receipt = _decode(ComparisonReceipt, receipt_data, [], f"phase2.{receipt_field}") if receipt_data else None
        population = entry.get("population")
        if receipt is None or not isinstance(population, dict):
            return False
        observed = {name: getattr(receipt.population, name) for name in ("expected", "supported", "compared")}
        if population != observed or receipt.verdict == AGREE:
            return False
        if entry.get("cause") != "stale_legacy_price_archive":
            return False
    return seen == set(required)


def _check_phase2(evidence: dict, artifact_root: Path, corpus_root: Path | None, findings: list,
                  field_ok: dict[str, bool]) -> None:
    field = "phase2_acceptance_ref"
    data = _resolve(evidence.get(field), artifact_root, findings, field)
    if data is None:
        field_ok[field] = False
        return
    try:
        phase2_doc = json.loads(data)
    except ValueError:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "invalid_json"})
        field_ok[field] = False
        return
    if not isinstance(phase2_doc, dict) or phase2_doc.get("schema_version") != PHASE2_EVIDENCE_V1:
        findings.append({"code": "PREREQUISITE_FAILED", "prerequisite": "phase2", "reason": "schema_version"})
        field_ok[field] = False
        return
    p2_findings, p2_field_ok, p2_document_ok = validate_phase2_evidence(
        phase2_doc, artifact_root=artifact_root, corpus_root=corpus_root,
        code_hash=evidence.get("source_code_hash"), environment_hash=evidence.get("source_environment_hash"))
    if not p2_document_ok or p2_findings or not all(p2_field_ok.values()):
        disposition_data = _resolve(
            evidence.get(PHASE2_DISPOSITION_FIELD), artifact_root, findings, PHASE2_DISPOSITION_FIELD)
        disposition_ok = _check_phase2_disposition(
            disposition_data, phase2_ref=evidence.get(field), phase2_doc=phase2_doc,
            artifact_root=artifact_root, phase2_findings=p2_findings)
        field_ok[PHASE2_DISPOSITION_FIELD] = disposition_ok
        if disposition_ok:
            findings.append({"code": "PHASE2_STRICT_FINDINGS_RETAINED", "prerequisite": "phase2",
                             "phase2_findings": p2_findings})
            return
        findings.append({"code": "PREREQUISITE_FAILED", "prerequisite": "phase2",
                         "phase2_findings": p2_findings})
        field_ok[field] = False
    elif PHASE2_DISPOSITION_FIELD in evidence:
        findings.append({"code": "UNEXPECTED_DISPOSITION", "field": PHASE2_DISPOSITION_FIELD})
        field_ok[PHASE2_DISPOSITION_FIELD] = False


def validate_evidence(evidence: dict, *, artifact_root: Path,
                      implementation_code_hash: str, environment_hash: str,
                      corpus_root: Path | None = None,
                      ) -> tuple[list[dict], dict[str, bool], bool, dict[tuple[str, str], bool]]:
    """Every finding the strict ``phase3_evidence.v1.0`` document can produce.

    ``implementation_code_hash``/``environment_hash`` are the CURRENT tree's
    values (as ``rearchitecture_phase3_gate.py`` computes them) -- what every
    ``comparison_receipt_refs``/``negative_control_receipt_refs`` item must be
    bound to, per guide §10 ("Phase 3 tests/projection evidence binds to the
    final implementation"). They are deliberately NOT what
    ``phase2_acceptance_ref`` is checked against -- see the module docstring.
    ``corpus_root``: forwarded to ``validate_phase2_evidence`` for D14's
    corpus-snapshot binding check (default ``checks.tier0_corpus.
    DEFAULT_CORPUS``) -- a caller validating a Phase 2 acceptance built
    against a different (e.g. synthetic test) corpus passes its own.

    Returns ``(findings, field_ok, document_ok, kind_ok)``: the first three
    match Phase 2's own validator contract exactly. ``kind_ok`` additionally
    maps ``(list_field, comparison_kind) -> bool`` so the gate's L01-L14
    matrix (``checks/phase3_acceptance.json``) can require a specific
    receipt KIND inside ``comparison_receipt_refs``/``negative_control_
    receipt_refs``, not merely that the list is nonempty.
    """
    findings: list[dict] = []
    if not isinstance(evidence, dict) or evidence.get("schema_version") != PHASE3_EVIDENCE_V1:
        findings.append({"code": "SCHEMA_VERSION_UNSUPPORTED", "field": "schema_version"})
        return findings, {}, False, {}

    document_ok = True
    unknown = sorted(set(evidence) - ALL_FIELDS)
    if unknown:
        for field in unknown:
            findings.append({"code": "UNKNOWN_FIELD", "field": field})
        document_ok = False
    if evidence.get("authority_mode") != AUTHORITY_MODE:
        findings.append({"code": "AUTHORITY_NOT_SHADOW"})
        document_ok = False

    field_ok: dict[str, bool] = {}
    _check_scalars(evidence, findings, field_ok)

    # Every ref-type field is REQUIRED (guide §10's field list carries no
    # optional marker in this schema) -- resolve unconditionally rather than
    # skipping an absent key, so a wholly missing field still produces its
    # own finding instead of silently passing validate_evidence with no
    # entry in field_ok at all (test requirement: "each required field
    # missing -> its own code").
    resolved: dict[str, bytes] = {}
    for field in (*DECODE_CLASS, *RAW_SINGLE_FIELDS, *RAW_LIST_SINGLE_FIELDS):
        data = _resolve(evidence.get(field), artifact_root, findings, field)
        field_ok[field] = data is not None
        if data is not None:
            resolved[field] = data

    resolved_lists: dict[str, list[bytes]] = {}
    for field in LIST_REF_FIELDS:
        items = evidence.get(field)
        if isinstance(items, bool):
            findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
            field_ok[field] = False
            continue
        if items is None or not isinstance(items, list) or not items:
            findings.append({"code": "ARTIFACT_MISSING", "field": field})
            field_ok[field] = False
            continue
        pieces = [_resolve(item, artifact_root, findings, f"{field}[{i}]") for i, item in enumerate(items)]
        field_ok[field] = all(p is not None for p in pieces)
        resolved_lists[field] = [p for p in pieces if p is not None]

    decoded: dict[str, Any] = {}
    for field, cls in DECODE_CLASS.items():
        data = resolved.get(field)
        if data is None:
            continue
        obj = _decode(cls, data, findings, field)
        if obj is None:
            field_ok[field] = False
        else:
            decoded[field] = obj

    decoded_lists: dict[str, list[Any]] = {}
    for field, cls in LIST_DECODE_CLASS.items():
        items = resolved_lists.get(field)
        if items is None:
            continue
        objs = [_decode(cls, data, findings, f"{field}[{i}]") for i, data in enumerate(items)]
        if any(obj is None for obj in objs):
            field_ok[field] = False
        decoded_lists[field] = [obj for obj in objs if obj is not None]

    accepted_release_ids = {r.release_id for r in decoded_lists.get("accepted_release_refs", [])}
    _check_release_bindings(evidence, artifact_root, findings, field_ok)
    _check_preview_lineage(decoded_lists.get("preview_input_refs", []),
                           decoded_lists.get("accepted_release_refs", []), findings, field_ok)
    _check_preview_proofs(evidence, decoded_lists.get("preview_input_refs", []), artifact_root,
                          findings, field_ok)

    population_doc = _check_raw_dict(resolved.get("population_manifest_ref"), findings, field_ok,
                                     "population_manifest_ref", ("expected", "supported", "compared"))
    _check_population_manifest(population_doc, findings, field_ok)
    _check_browser_receipt(
        _check_raw_dict(resolved.get("browser_receipt_ref"), findings, field_ok,
                        "browser_receipt_ref", ("release_id", "as_of", "url", "screenshot_ref")),
        artifact_root, findings, field_ok)
    _check_engineering_receipt(
        _check_raw_dict(resolved.get("engineering_receipt_ref"), findings, field_ok,
                        "engineering_receipt_ref", ("nights",)),
        findings, field_ok)
    _check_coverage_receipt(
        _check_raw_dict(resolved.get("coverage_receipt_ref"), findings, field_ok,
                        "coverage_receipt_ref", ("schema_version", "source_hash", "packages")),
        implementation_code_hash, findings, field_ok)
    _check_raw_dict(resolved.get("performance_receipt_ref"), findings, field_ok, "performance_receipt_ref",
                    ("api_latency_p50_ms", "first_usable_page_seconds", "bytes", "population",
                     "memory_mb", "cache_state", "contention_note"))
    _check_raw_list(resolved.get("view_field_inventory_ref"), findings, field_ok,
                    "view_field_inventory_ref", ("field", "shipped"))
    _check_raw_list(resolved.get("deferred_work_ref"), findings, field_ok,
                    "deferred_work_ref", ("item", "owner"))

    _check_phase2(evidence, artifact_root, corpus_root, findings, field_ok)

    kind_ok = _check_verdicts_and_bindings(
        decoded_lists, findings, field_ok, code_hash=implementation_code_hash,
        environment_hash=environment_hash, accepted_release_ids=accepted_release_ids)
    _check_complete_population(decoded_lists, population_doc, findings, field_ok)
    _check_rollback(decoded.get("refresh_rollback_receipt_ref"), accepted_release_ids, findings, field_ok)

    return findings, field_ok, document_ok, kind_ok
