#!/usr/bin/env python3
"""Builds a ``phase3_evidence.v1.0`` manifest and runs the strict validator on it.

Mirrors ``checks/rearchitecture_phase2_evidence_build.py``: every field this
script can compute or copy from an ALREADY-PRODUCED local file is filled in;
every field whose real producer has not run yet is left ABSENT, never faked
with a placeholder. Run the strict validator (via ``rearchitecture_phase3_
gate.py``) afterwards to see exactly which items are still missing --
``MISSING_EVIDENCE`` per field/row, never a silent pass.

Auto-resolved without any flag:

- ``implementation_code_hash`` -- ``rearchitecture_phase1_gate.source_hash``
  over the current tree (same whole-tree hash Phase 1/2 use as ``code_hash``).
- ``environment_hash`` -- ``rearchitecture_phase2_gate.environment_hash``
  (reused, never reimplemented).
- ``frontend_lock_hash`` -- content hash of ``ui/package-lock.json`` when
  present on disk (read-only; this script never writes under ``ui/``).
- ``authority_mode`` -- always ``"shadow"`` (guide §10 fixes it).

``--phase2-evidence``/``--phase2-artifact-root`` copy Phase 2's OWN evidence
document into ``<artifact-root>/phase2/_evidence.json`` and every artifact it
references into the shared artifact root at its unchanged relative path. This
lets the Phase 3 validator resolve Phase 2's nested refs without rewriting
their hashes or paths -- see ``rearchitecture_
phase3_evidence.py``'s module docstring for why no separate root field exists
in the schema. ``--source-code-hash``/``--source-environment-hash`` default
to the copied Phase2Evidence document's OWN declared ``code_hash``/
``environment_hash`` (its producer commit) when omitted -- overriding them is
only for a caller deliberately testing a mismatch.

Repeatable receipt flags take ``KIND=PATH`` (``--comparison-receipt
bridge_identity_parity=/path/to/receipt.json``); the kind becomes the
receipt's declared ``comparison_kind`` inside ``checks/phase3_acceptance.json``'s
L01-L14 matrix, never invented by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import rearchitecture_phase2_evidence as p2evidence  # noqa: E402
from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from checks.rearchitecture_phase3_evidence import PHASE3_EVIDENCE_V1, validate_evidence  # noqa: E402

_LIST_FIELDS = ("preview_input_refs", "accepted_release_refs",
                "comparison_receipt_refs", "negative_control_receipt_refs")
_SINGLE_REF_FIELDS = ("population_manifest_ref", "browser_receipt_ref",
                      "refresh_rollback_receipt_ref", "engineering_receipt_ref",
                      "coverage_receipt_ref", "performance_receipt_ref",
                      "view_field_inventory_ref", "deferred_work_ref")


def _publish(path: Path, artifact_root: Path, name: str) -> dict:
    data = path.read_bytes()
    dest = artifact_root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"path": name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def _copy_phase2(phase2_evidence_path: Path, phase2_artifact_root: Path,
                 artifact_root: Path) -> tuple[dict, dict]:
    """Copies Phase 2's evidence document and every ref it names into
    ``<artifact_root>/phase2/...``, preserving relative paths. Returns
    ``(phase2_acceptance_ref, phase2_doc)``."""
    doc = json.loads(phase2_evidence_path.read_text())
    def copy_one(rel: str) -> None:
        src = phase2_artifact_root / rel
        if not src.is_file():
            return
        dest = artifact_root / rel
        try:
            dest.resolve().relative_to(artifact_root.resolve())
        except ValueError:
            raise ValueError("Phase 2 artifact path escapes artifact root")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    for field in p2evidence.REF_FIELDS:
        ref = doc.get(field)
        if isinstance(ref, dict) and isinstance(ref.get("path"), str):
            copy_one(ref["path"])
    for field in p2evidence.LIST_REF_FIELDS:
        for ref in doc.get(field) or []:
            if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                copy_one(ref["path"])

    phase2_doc_ref = _publish(phase2_evidence_path, artifact_root, "phase2/_evidence.json")
    return phase2_doc_ref, doc


def _publish_browser_receipt(path: Path, artifact_root: Path) -> dict:
    data = path.read_bytes()
    document = json.loads(data)
    screenshot_ref = document.get("screenshot_ref") if isinstance(document, dict) else None
    if not isinstance(screenshot_ref, dict) or not isinstance(screenshot_ref.get("path"), str):
        return _publish(path, artifact_root, "browser_receipt_ref.json")
    screenshot_path = (path.parent / screenshot_ref["path"]).resolve()
    try:
        screenshot_path.relative_to(path.parent.resolve())
    except ValueError:
        raise ValueError("browser screenshot path escapes receipt directory")
    if not screenshot_path.is_file():
        raise ValueError("browser screenshot referenced by receipt is missing")
    screenshot = _publish(screenshot_path, artifact_root, "browser_screenshot.png")
    document["screenshot_ref"] = screenshot
    receipt = json.dumps(document, indent=2, sort_keys=True).encode()
    receipt_path = artifact_root / "browser_receipt_ref.json"
    receipt_path.write_bytes(receipt)
    return {"path": receipt_path.name, "content_hash": "sha256:" + hashlib.sha256(receipt).hexdigest()}


def _parse_kind_path(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items or []:
        kind, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected KIND=PATH, got {item!r}")
        out.append((kind, Path(path)))
    return out


def build(*, artifact_root: Path, phase2_evidence: Path | None, phase2_artifact_root: Path | None,
         source_code_hash: str | None, source_environment_hash: str | None,
         mapping_version: str | None, preview_inputs: list[Path], accepted_releases: list[str],
         population_manifest: Path | None, comparison_receipts: list[str],
         negative_control_receipts: list[str], browser_receipt: Path | None,
         refresh_rollback_receipt: Path | None, engineering_receipt: Path | None,
         coverage_receipt: Path | None, performance_receipt: Path | None,
         view_field_inventory: Path | None, deferred_work: Path | None,
         ui_lock: Path | None, phase2_handoff_disposition: Path | None = None) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    evidence: dict = {
        "schema_version": PHASE3_EVIDENCE_V1,
        "authority_mode": "shadow",
        "implementation_code_hash": source_hash(source_files(ROOT)),
    }
    evidence["environment_hash"], _source = _environment_hash(ROOT)
    if ui_lock is None:
        ui_lock = ROOT / "ui" / "package-lock.json"
    if ui_lock.is_file():
        evidence["frontend_lock_hash"] = "sha256:" + hashlib.sha256(ui_lock.read_bytes()).hexdigest()

    if phase2_evidence is not None and phase2_artifact_root is not None:
        phase2_ref, phase2_doc = _copy_phase2(phase2_evidence, phase2_artifact_root, artifact_root)
        evidence["phase2_acceptance_ref"] = phase2_ref
        evidence["source_code_hash"] = source_code_hash or phase2_doc.get("code_hash")
        evidence["source_environment_hash"] = source_environment_hash or phase2_doc.get("environment_hash")
    elif source_code_hash and source_environment_hash:
        evidence["source_code_hash"] = source_code_hash
        evidence["source_environment_hash"] = source_environment_hash

    if phase2_handoff_disposition is not None:
        evidence["phase2_handoff_disposition_ref"] = _publish(
            phase2_handoff_disposition, artifact_root, "phase2_handoff_disposition.json")

    if mapping_version:
        evidence["mapping_version"] = mapping_version

    if preview_inputs:
        evidence["preview_input_refs"] = [
            _publish(p, artifact_root, f"preview_input_{i}.json") for i, p in enumerate(preview_inputs)]
    if accepted_releases:
        # RELEASE_PATH=BINDING_PATH: each accepted release binds to its own
        # real projection_binding.v1.0 document (P3-1c, coordinator
        # instruction 2026-09-14) -- see rearchitecture_phase3_evidence.py's
        # "Each accepted release binds to its own real projection_binding..."
        # section. A release with no real binding yet is not an accepted
        # release this builder can emit; omit it rather than fake the pair.
        refs = []
        for i, (release_path, binding_path) in enumerate(_parse_kind_path(accepted_releases)):
            ref = _publish(Path(release_path), artifact_root, f"accepted_release_{i}.json")
            ref["binding_ref"] = _publish(binding_path, artifact_root, f"accepted_release_{i}_binding.json")
            refs.append(ref)
        evidence["accepted_release_refs"] = refs
    if population_manifest is not None:
        evidence["population_manifest_ref"] = _publish(
            population_manifest, artifact_root, "population_manifest.json")

    comparisons = _parse_kind_path(comparison_receipts)
    if comparisons:
        evidence["comparison_receipt_refs"] = [
            _publish(path, artifact_root, f"comparison_{kind}.json") for kind, path in comparisons]
    negatives = _parse_kind_path(negative_control_receipts)
    if negatives:
        evidence["negative_control_receipt_refs"] = [
            _publish(path, artifact_root, f"negative_control_{kind}.json") for kind, path in negatives]

    if browser_receipt is not None:
        evidence["browser_receipt_ref"] = _publish_browser_receipt(browser_receipt, artifact_root)
    single = {"refresh_rollback_receipt_ref": refresh_rollback_receipt,
             "engineering_receipt_ref": engineering_receipt, "coverage_receipt_ref": coverage_receipt,
             "performance_receipt_ref": performance_receipt, "view_field_inventory_ref": view_field_inventory,
             "deferred_work_ref": deferred_work}
    for field, path in single.items():
        if path is not None:
            evidence[field] = _publish(path, artifact_root, f"{field}.json")

    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--phase2-evidence", type=Path)
    parser.add_argument("--phase2-artifact-root", type=Path)
    parser.add_argument("--source-code-hash")
    parser.add_argument("--source-environment-hash")
    parser.add_argument("--phase2-handoff-disposition", type=Path,
                        help="private disposition bound to this exact Phase 2 evidence document")
    parser.add_argument("--mapping-version")
    parser.add_argument("--preview-input", action="append", type=Path, default=[])
    parser.add_argument("--accepted-release", action="append", default=[],
                        metavar="RELEASE_PATH=BINDING_PATH",
                        help="a PreviewRelease document and its own real projection_binding.v1.0 "
                             "document (engine.v2.serving.projections.projection_binding), paired")
    parser.add_argument("--population-manifest", type=Path)
    parser.add_argument("--comparison-receipt", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument("--negative-control-receipt", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument("--browser-receipt", type=Path)
    parser.add_argument("--refresh-rollback-receipt", type=Path)
    parser.add_argument("--engineering-receipt", type=Path)
    parser.add_argument("--coverage-receipt", type=Path)
    parser.add_argument("--performance-receipt", type=Path)
    parser.add_argument("--view-field-inventory", type=Path)
    parser.add_argument("--deferred-work", type=Path)
    parser.add_argument("--ui-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = build(
        artifact_root=args.artifact_root, phase2_evidence=args.phase2_evidence,
        phase2_artifact_root=args.phase2_artifact_root, source_code_hash=args.source_code_hash,
        source_environment_hash=args.source_environment_hash, mapping_version=args.mapping_version,
        preview_inputs=args.preview_input, accepted_releases=args.accepted_release,
        population_manifest=args.population_manifest, comparison_receipts=args.comparison_receipt,
        negative_control_receipts=args.negative_control_receipt, browser_receipt=args.browser_receipt,
        refresh_rollback_receipt=args.refresh_rollback_receipt, engineering_receipt=args.engineering_receipt,
        coverage_receipt=args.coverage_receipt, performance_receipt=args.performance_receipt,
        view_field_inventory=args.view_field_inventory, deferred_work=args.deferred_work,
        ui_lock=args.ui_lock, phase2_handoff_disposition=args.phase2_handoff_disposition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    findings, field_ok, document_ok, kind_ok = validate_evidence(
        evidence, artifact_root=args.artifact_root,
        implementation_code_hash=evidence["implementation_code_hash"],
        environment_hash=evidence["environment_hash"])
    print(json.dumps({"document_ok": document_ok, "field_ok": field_ok,
                      "kind_ok": {"|".join(k): v for k, v in kind_ok.items()},
                      "findings": findings}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
