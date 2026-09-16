#!/usr/bin/env python3
"""Build one source-verified Phase 3 PreviewInput from a delivered Phase 2 release."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import ObjectRef, PreviewInput
from engine.v2.foundation import SystemClock, content_hash, from_document, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.serving.legacy_bundle import load_legacy_bundle


def _artifact(conn, artifact_id: str):
    row = conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown artifact: {artifact_id}")
    return json.loads(row["ref_json"])


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--score-artifact-id", required=True)
    parser.add_argument("--bundle-artifact-id", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--snapshot-artifact-id", required=True)
    parser.add_argument("--materialization-request-artifact-id", required=True)
    parser.add_argument("--finality-artifact-id", required=True)
    parser.add_argument("--model-evidence-artifact-id", required=True)
    parser.add_argument("--score-comparison-receipt-ref", required=True)
    parser.add_argument("--render-comparison-receipt-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        release_row = conn.execute("SELECT * FROM releases WHERE release_id = ?", (args.release_id,)).fetchone()
        if release_row is None or release_row["published_at"] is None or release_row["delivered_at"] is None:
            raise ValueError("source release is not delivered")
        manifest = json.loads(release_row["manifest_json"])
        score_ref = _artifact(conn, args.score_artifact_id)
        bundle_ref = _artifact(conn, args.bundle_artifact_id)
        snapshot_artifact = _artifact(conn, args.snapshot_artifact_id)
        request_ref = _artifact(conn, args.materialization_request_artifact_id)
        finality_ref = _artifact(conn, args.finality_artifact_id)
        model_evidence_ref = _artifact(conn, args.model_evidence_artifact_id)
        if manifest.get("files", {}).get("bundle.tar", {}).get("artifact_id") != args.bundle_artifact_id:
            raise ValueError("source bundle is not the delivered release bundle")
        _bundle_rows, bundle_manifest = load_legacy_bundle(args.bundle_dir)
        request_path = args.store_root / request_ref["storage_key"]
        request = json.loads(request_path.read_text())
        snapshot = json.loads((args.store_root / snapshot_artifact["storage_key"]).read_text())
        if request.get("snapshot_ref", {}).get("snapshot_id") != snapshot.get("snapshot_id"):
            raise ValueError("materialization request and snapshot artifact disagree")
        score_job = conn.execute(
            "SELECT spec_json FROM jobs WHERE job_id IN (SELECT job_id FROM attempts WHERE attempt_id IN "
            "(SELECT attempt_id FROM attempt_outputs WHERE artifact_id = ?))", (args.score_artifact_id,)).fetchone()
        if score_job is None:
            raise ValueError("score artifact has no producing job")
        score_spec = json.loads(score_job["spec_json"])
        score_path = args.store_root / score_ref["storage_key"]
        score_doc = json.loads(score_path.read_text())
        legacy_snapshot = from_document(ObjectRef, request["legacy_snapshot_object_ref"])
        preview = PreviewInput(
            source_release_id=args.release_id,
            source_release_manifest_ref=release_row["manifest_hash"],
            snapshot_ref=snapshot["snapshot_id"],
            legacy_snapshot_object_ref=legacy_snapshot,
            score_batch_ref=score_ref["content_hash"],
            score_job_input_refs=tuple(score_spec["input_refs"]),
            bundle_manifest_ref=content_hash(bundle_manifest),
            model_registry_artifact_refs=tuple(request["registry_and_model_refs"]),
            model_evidence_ref=model_evidence_ref["content_hash"],
            finality_ref=finality_ref["content_hash"],
            expected_population_ref=content_hash(score_doc["expected_population"]),
            score_comparison_receipt_ref=args.score_comparison_receipt_ref,
            render_comparison_receipt_ref=args.render_comparison_receipt_ref,
            source_code_hash=score_spec["implementation_ref"],
            source_environment_hash=score_spec["environment_ref"],
        )
        provenance = {
            "schema_version": "phase3_source_provenance.v1.0",
            "release_id": args.release_id,
            "release_manifest_hash": release_row["manifest_hash"],
            "delivered_at": release_row["delivered_at"],
            "score_artifact": score_ref,
            "bundle_artifact": bundle_ref,
            "snapshot_artifact": snapshot_artifact,
            "materialization_request_artifact": request_ref,
            "finality_artifact": finality_ref,
            "model_evidence_artifact": model_evidence_ref,
            "expected_population": len(score_doc["expected_population"]),
        }
    finally:
        conn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_document(preview), indent=2, sort_keys=True) + "\n")
    args.output.with_name("source_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preview_input": str(args.output), "provenance": str(args.output.with_name("source_provenance.json")), "population": provenance["expected_population"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
