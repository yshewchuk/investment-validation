#!/usr/bin/env python3
"""Builds a ``phase2_evidence.v1.0`` manifest and runs the strict validator on it.

Auto-resolves from the catalog at ``--root``: the head snapshot for
``--scope``, its pinned legacy snapshot object ref, its committed import
receipts, and (when ``--score-receipt`` is given) the D15 snapshot job's
dependency plans, read via that receipt's own ``right_ref`` (the snapshot
job id ``checks/rearchitecture_phase2_parity.py`` recorded). Every other
receipt (render/corpus/rollback, and the fault matrix) is supplied as an
already-produced local file and copied verbatim under ``--artifact-root``.

Refs not supplied on the command line stay ABSENT from the evidence document
-- never filled with a placeholder -- so the strict validator reports
``MISSING_EVIDENCE`` for exactly the rows that still need them (phase-2 guide
§12.1: "It does not accept a summary boolean in place of the referenced
receipts.").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_evidence import validate_evidence  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.contracts import SnapshotImportReceipt  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.legacy_mapping import build_legacy_mapping  # noqa: E402
from engine.v2.data.reference_catalog import pinned_materialization_refs, reference_inputs_for_snapshot  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import fail  # noqa: E402
from engine.v2.ops.input_bindings import recorded_bindings  # noqa: E402
from engine.v2.ops.snapshot_stages import request_from_artifact  # noqa: E402

SCHEMA_VERSION = "phase2_evidence.v1.0"
_RECEIPT_FLAGS = {"score_receipt": "comparison_receipt_ref", "render_receipt": "render_comparison_receipt_ref",
                 "corpus_receipt": "corpus_comparison_receipt_ref", "rollback_receipt": "rollback_receipt_ref",
                 "fault_matrix": "fault_matrix_ref"}


def _publish_bytes(data: bytes, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / name).write_bytes(data)
    return {"path": name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def _publish_document(value, artifact_root: Path, name: str) -> dict:
    return _publish_bytes(json.dumps(to_document(value), indent=2, sort_keys=True).encode(),
                          artifact_root, name)


def _head_snapshot_id(conn, scope: str) -> str | None:
    row = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope=?",
                       (scope,)).fetchone()
    return row[0] if row else None


def _committed_import_receipts(conn, scope: str) -> list[SnapshotImportReceipt]:
    rows = conn.execute(
        "SELECT receipt_id, attempt_id, fence, source_manifest_hash, result_snapshot_id "
        "FROM data_import_receipts WHERE scope=? AND status='committed' ORDER BY registered_at",
        (scope,)).fetchall()
    return [SnapshotImportReceipt(
        receipt_id=r["receipt_id"], request_hash=r["source_manifest_hash"], attempt_id=r["attempt_id"],
        fence=r["fence"], snapshot_ref=Repository(conn).resolve(r["result_snapshot_id"]),
        legacy_snapshot_object_ref=None, prior_head_snapshot_id=None,
        resulting_head_snapshot_id=r["result_snapshot_id"], resulting_head_generation=None,
        status="committed", problem=None, envelope={}) for r in rows]


def _dependency_plan_refs(conn, store, score_doc: dict, artifact_root: Path) -> list[dict]:
    """The D15 snapshot job's materialization request, read from ITS OWN
    ``right_ref`` (the snapshot job id ``score_parity_receipt`` recorded),
    turned into one ``DependencyPlan`` per query via ``explain_dependencies``.
    """
    snapshot_job_id = score_doc["right_ref"]
    attempt = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (snapshot_job_id,)).fetchone()
    if attempt is None:
        raise fail("INPUT_CHANGED", "the score receipt's snapshot job has no succeeded attempt")
    binding = recorded_bindings(conn, attempt[0]).get("materialization_request.json")
    if binding is None:
        raise fail("INPUT_CHANGED", "the score receipt's snapshot job bound no materialization request")
    request = request_from_artifact(conn, store, binding.artifact_id)
    repo = Repository(conn, store)
    refs = []
    for index, (table_name, query) in enumerate(sorted(request.table_queries.items())):
        plan = repo.explain_dependencies(query, table_name=table_name)
        refs.append(_publish_document(plan, artifact_root, f"dependency_plan_{index}_{table_name}.json"))
    return refs


def _catalog_fields(conn, store, scope: str, artifact_root: Path) -> dict:
    """Everything auto-resolved from the catalog: never gated by a CLI flag."""
    fields: dict = {}
    head_id = _head_snapshot_id(conn, scope)
    if head_id is None:
        return fields
    snapshot = Repository(conn).resolve(head_id)
    fields["snapshot_ref"] = _publish_document(snapshot, artifact_root, "snapshot_ref.json")
    try:
        pinned = pinned_materialization_refs(
            reference_inputs_for_snapshot(conn, scope=scope, snapshot_id=head_id))
        fields["legacy_snapshot_object_ref"] = _publish_document(
            pinned["legacy_snapshot_object_ref"], artifact_root, "legacy_snapshot_object_ref.json")
    except DataError:
        pass  # no pinned reference inputs recorded yet for this head snapshot
    imports = _committed_import_receipts(conn, scope)
    if imports:
        fields["import_receipt_refs"] = [
            _publish_document(r, artifact_root, f"import_receipt_{i}.json") for i, r in enumerate(imports)]
    fields["table_contract_mapping_hash"] = content_hash(build_legacy_mapping())
    return fields


def build(root: Path, *, scope: str, artifact_root: Path, score_receipt: Path | None,
         render_receipt: Path | None, corpus_receipt: Path | None, rollback_receipt: Path | None,
         fault_matrix: Path | None) -> dict:
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        store = ArtifactStore(root)
        evidence = {"schema_version": SCHEMA_VERSION, "authority_mode": "shadow",
                   "code_hash": source_hash(source_files(ROOT))}
        evidence["environment_hash"], _source = _environment_hash(ROOT)
        evidence.update(_catalog_fields(conn, store, scope, artifact_root))
        supplied = {"score_receipt": score_receipt, "render_receipt": render_receipt,
                   "corpus_receipt": corpus_receipt, "rollback_receipt": rollback_receipt,
                   "fault_matrix": fault_matrix}
        for flag, path in supplied.items():
            if path is None:
                continue
            data = path.read_bytes()
            evidence[_RECEIPT_FLAGS[flag]] = _publish_bytes(data, artifact_root, path.name)
        if score_receipt is not None:
            score_doc = json.loads(score_receipt.read_bytes())
            pop = score_doc.get("population", {})
            for pop_key, ev_key in (("expected", "expected_population"), ("supported", "supported_population"),
                                    ("compared", "compared_population")):
                if pop.get(pop_key) is not None:
                    evidence[ev_key] = pop[pop_key]
            evidence["dependency_plan_refs"] = _dependency_plan_refs(conn, store, score_doc, artifact_root)
    finally:
        conn.close()
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--score-receipt", type=Path)
    parser.add_argument("--render-receipt", type=Path)
    parser.add_argument("--corpus-receipt", type=Path)
    parser.add_argument("--rollback-receipt", type=Path)
    parser.add_argument("--fault-matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = build(args.root, scope=args.scope, artifact_root=args.artifact_root,
                     score_receipt=args.score_receipt, render_receipt=args.render_receipt,
                     corpus_receipt=args.corpus_receipt, rollback_receipt=args.rollback_receipt,
                     fault_matrix=args.fault_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    findings, field_ok, document_ok = validate_evidence(
        evidence, artifact_root=args.artifact_root, code_hash=evidence["code_hash"],
        environment_hash=evidence["environment_hash"])
    print(json.dumps({"document_ok": document_ok, "field_ok": field_ok, "findings": findings},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
