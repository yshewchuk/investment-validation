#!/usr/bin/env python3
"""Build one source-verified Phase 3 PreviewInput from a delivered Phase 2 release."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import ArtifactRef, ObjectRef, PreviewInput
from engine.v2.diagnosis.receipt import ComparisonReceipt
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, from_document, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.serving.legacy_bundle import load_legacy_bundle


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _artifact(conn, store: ArtifactStore, artifact_id: str, *, schema: str | None = None):
    row = conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown artifact: {artifact_id}")
    ref = from_document(ArtifactRef, json.loads(row["ref_json"]))
    if schema is not None and ref.schema_ref != schema:
        raise ValueError(f"{artifact_id}: expected {schema}, got {ref.schema_ref}")
    return ref, store.read_verified(ref)


def _producer_kind(conn, artifact_id: str) -> str:
    rows = conn.execute(
        "SELECT DISTINCT jobs.kind FROM attempt_outputs JOIN attempts USING(attempt_id) JOIN jobs USING(job_id) "
        "WHERE attempt_outputs.artifact_id = ?", (artifact_id,)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"{artifact_id}: artifact has no producing job")
    return rows[0]["kind"]


def _producer_job(conn, artifact_id: str) -> str:
    rows = conn.execute(
        "SELECT DISTINCT jobs.job_id FROM attempt_outputs JOIN attempts USING(attempt_id) JOIN jobs USING(job_id) "
        "WHERE attempt_outputs.artifact_id = ? AND attempts.state = 'succeeded' AND jobs.state = 'succeeded'",
        (artifact_id,)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"{artifact_id}: no unique succeeded producer job")
    return rows[0]["job_id"]


def _require_render_chain(conn, render_job_id: str, render_artifact_id: str, *, score_artifact_id: str,
                          finality_artifact_id: str, model_artifact_id: str) -> dict:
    row = conn.execute("SELECT kind, state, spec_json FROM jobs WHERE job_id = ?", (render_job_id,)).fetchone()
    if row is None or row["kind"] != "legacy_render" or row["state"] != "succeeded":
        raise ValueError("render job is not a succeeded legacy_render job")
    output = conn.execute(
        "SELECT artifact_id FROM attempt_outputs JOIN attempts USING(attempt_id) WHERE attempts.job_id = ? "
        "AND attempts.state = 'succeeded' AND name = 'legacy_render'", (render_job_id,)).fetchall()
    if {item["artifact_id"] for item in output} != {render_artifact_id}:
        raise ValueError("render artifact is not the succeeded render job output")
    attempts = conn.execute("SELECT attempt_id FROM attempts WHERE job_id = ? AND state = 'succeeded'",
                            (render_job_id,)).fetchall()
    expected = {"score.json": score_artifact_id, "finality.json": finality_artifact_id,
                "model_evidence.json": model_artifact_id}
    for attempt in attempts:
        rows = conn.execute("SELECT name, artifact_id FROM attempt_input_bindings WHERE attempt_id = ?",
                            (attempt["attempt_id"],)).fetchall()
        bindings = {item["name"]: item["artifact_id"] for item in rows}
        if all(bindings.get(name) == artifact_id for name, artifact_id in expected.items()):
            return {"attempt_id": attempt["attempt_id"], "bindings": bindings}
    raise ValueError("render attempt does not record the supplied score/finality/model artifacts")


def _require_score_chain(conn, score_job_id: str, score_artifact_id: str, *, snapshot_artifact_id: str,
                         request_artifact_id: str, finality_artifact_id: str) -> dict:
    row = conn.execute("SELECT kind, state, spec_json FROM jobs WHERE job_id = ?", (score_job_id,)).fetchone()
    if row is None or row["kind"] != "legacy_score" or row["state"] != "succeeded":
        raise ValueError("D15 score job is not a succeeded legacy_score job")
    attempts = conn.execute("SELECT attempt_id FROM attempts WHERE job_id = ? AND state = 'succeeded'",
                            (score_job_id,)).fetchall()
    expected = {"snapshot_ref.json": snapshot_artifact_id,
                "materialization_request.json": request_artifact_id,
                "finality.json": finality_artifact_id}
    for attempt in attempts:
        output = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id = ? AND name = 'legacy_score'",
                              (attempt["attempt_id"],)).fetchone()
        rows = conn.execute("SELECT name, artifact_id FROM attempt_input_bindings WHERE attempt_id = ?",
                            (attempt["attempt_id"],)).fetchall()
        bindings = {item["name"]: item["artifact_id"] for item in rows}
        if output is not None and output["artifact_id"] == score_artifact_id and all(
                bindings.get(name) == artifact_id for name, artifact_id in expected.items()):
            return {"attempt_id": attempt["attempt_id"], "bindings": bindings}
    raise ValueError("D15 score attempt does not record supplied score/snapshot/request/finality")


def _bundle_manifest_from_artifact(data: bytes) -> dict:
    """Read the delivered tar without trusting caller supplied bundle bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                destination = (root / member.name).resolve()
                if not member.isfile() or root not in destination.parents:
                    raise ValueError("delivered bundle contains an unsafe member")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("delivered bundle member is unreadable")
                destination.write_bytes(source.read())
        bundle_root = root / "bundle" if (root / "bundle").is_dir() else root
        _rows, manifest = load_legacy_bundle(bundle_root)
        return manifest


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
    parser.add_argument("--score-comparison-receipt", type=Path, required=True)
    parser.add_argument("--render-comparison-receipt", type=Path, required=True)
    parser.add_argument("--render-job-id", required=True)
    parser.add_argument("--render-artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    store = ArtifactStore(args.store_root)
    try:
        release_row = conn.execute("SELECT * FROM releases WHERE release_id = ?", (args.release_id,)).fetchone()
        if release_row is None or release_row["published_at"] is None or release_row["delivered_at"] is None:
            raise ValueError("source release is not delivered")
        manifest = json.loads(release_row["manifest_json"])
        score_ref, score_bytes = _artifact(conn, store, args.score_artifact_id, schema="legacy_action.v1.0")
        bundle_ref, bundle_bytes = _artifact(conn, store, args.bundle_artifact_id, schema="legacy_action.v1.0")
        snapshot_artifact, snapshot_bytes = _artifact(conn, store, args.snapshot_artifact_id, schema="snapshot_ref.v1.0")
        request_ref, request_bytes = _artifact(conn, store, args.materialization_request_artifact_id, schema="legacy_materialization_request.v1.0")
        finality_ref, finality_bytes = _artifact(conn, store, args.finality_artifact_id, schema="legacy_action.v1.0")
        model_evidence_ref, model_evidence_bytes = _artifact(conn, store, args.model_evidence_artifact_id, schema="legacy_action.v1.0")
        render_ref, render_bytes = _artifact(conn, store, args.render_artifact_id, schema="legacy_action.v1.0")
        if args.render_artifact_id != args.bundle_artifact_id:
            raise ValueError("render output is not the delivered bundle artifact")
        typed_manifest = dict(manifest)
        typed_manifest["files"] = {name: from_document(ArtifactRef, ref)
                                   for name, ref in manifest.get("files", {}).items()}
        if content_hash(typed_manifest) != release_row["manifest_hash"]:
            raise ValueError("source release manifest hash does not match retained manifest")
        expected_kinds = ((args.score_artifact_id, "legacy_score"),
                          (args.finality_artifact_id, "legacy_finality"),
                          (args.model_evidence_artifact_id, "legacy_model_evidence"))
        for artifact_id, expected_kind in expected_kinds:
            if _producer_kind(conn, artifact_id) != expected_kind:
                raise ValueError(f"{artifact_id}: wrong producer kind")
        if _producer_kind(conn, args.render_artifact_id) != "legacy_render":
            raise ValueError("render artifact has the wrong producer kind")
        render_chain = _require_render_chain(conn, args.render_job_id, args.render_artifact_id,
                              score_artifact_id=args.score_artifact_id,
                              finality_artifact_id=args.finality_artifact_id,
                              model_artifact_id=args.model_evidence_artifact_id)
        if manifest.get("files", {}).get("bundle.tar", {}).get("artifact_id") != args.bundle_artifact_id:
            raise ValueError("source bundle is not the delivered release bundle")
        _bundle_rows, bundle_manifest = load_legacy_bundle(args.bundle_dir)
        if content_hash(_bundle_manifest_from_artifact(bundle_bytes)) != content_hash(bundle_manifest):
            raise ValueError("bundle directory does not match the delivered bundle artifact")
        request = json.loads(request_bytes)
        snapshot = json.loads(snapshot_bytes)
        if request.get("snapshot_ref", {}).get("snapshot_id") != snapshot.get("snapshot_id"):
            raise ValueError("materialization request and snapshot artifact disagree")
        score_receipt_bytes = args.score_comparison_receipt.read_bytes()
        render_receipt_bytes = args.render_comparison_receipt.read_bytes()
        score_receipt = from_document(ComparisonReceipt, json.loads(score_receipt_bytes))
        render_receipt = from_document(ComparisonReceipt, json.loads(render_receipt_bytes))
        if score_receipt.comparison_kind != "score_record_parity":
            raise ValueError("score comparison receipt has the wrong kind")
        if render_receipt.comparison_kind != "render_bundle_parity":
            raise ValueError("render comparison receipt has the wrong kind")
        score_job = conn.execute("SELECT job_id, spec_json FROM jobs WHERE job_id = ?", (score_receipt.right_ref,)).fetchone()
        if score_job is None:
            raise ValueError("D15 score receipt names an unknown score job")
        score_chain = _require_score_chain(conn, score_job["job_id"], args.score_artifact_id,
            snapshot_artifact_id=args.snapshot_artifact_id,
            request_artifact_id=args.materialization_request_artifact_id,
            finality_artifact_id=args.finality_artifact_id)
        score_spec = json.loads(score_job["spec_json"])
        score_doc = json.loads(score_bytes)
        if (score_receipt.verdict != "agree" or render_receipt.verdict != "agree"
                or score_receipt.envelope.snapshot_id != snapshot["snapshot_id"]
                or render_receipt.envelope.snapshot_id != snapshot["snapshot_id"]
                or score_receipt.right_ref != score_job["job_id"]
                or render_receipt.right_ref != args.render_job_id):
            raise ValueError("comparison receipts are not bound to the source jobs and snapshot")
        if args.score_comparison_receipt_ref != _sha256(score_receipt_bytes) or \
                args.render_comparison_receipt_ref != _sha256(render_receipt_bytes):
            raise ValueError("comparison receipt refs do not name the supplied receipt bytes")
        legacy_snapshot = from_document(ObjectRef, request["legacy_snapshot_object_ref"])
        preview = PreviewInput(
            source_release_id=args.release_id,
            source_release_manifest_ref=release_row["manifest_hash"],
            snapshot_ref=snapshot["snapshot_id"],
            legacy_snapshot_object_ref=legacy_snapshot,
            score_batch_ref=score_ref.content_hash,
            score_job_input_refs=tuple(score_spec["input_refs"]),
            bundle_manifest_ref=content_hash(bundle_manifest),
            model_registry_artifact_refs=tuple(request["registry_and_model_refs"]),
            model_evidence_ref=model_evidence_ref.content_hash,
            finality_ref=finality_ref.content_hash,
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
            "score_artifact": to_document(score_ref),
            "bundle_artifact": to_document(bundle_ref),
            "snapshot_artifact": to_document(snapshot_artifact),
            "materialization_request_artifact": to_document(request_ref),
            "finality_artifact": to_document(finality_ref),
            "model_evidence_artifact": to_document(model_evidence_ref),
            "render_artifact": to_document(render_ref),
            "score_job_id": score_job["job_id"],
            "score_attempt_id": score_chain["attempt_id"],
            "score_attempt_bindings": score_chain["bindings"],
            "render_job_id": args.render_job_id,
            "render_attempt_id": render_chain["attempt_id"],
            "render_attempt_bindings": render_chain["bindings"],
            "bundle_artifact_bytes_hash": _sha256(bundle_bytes),
            "bundle_manifest_ref": content_hash(bundle_manifest),
            "release_manifest": manifest,
            "score_spec": score_spec,
            "score_job_input_refs": list(score_spec["input_refs"]),
            "model_registry_artifact_refs": list(request["registry_and_model_refs"]),
            "score_comparison_receipt": {"path": "receipts/score_comparison.json",
                                                "content_hash": _sha256(score_receipt_bytes)},
            "render_comparison_receipt": {"path": "receipts/render_comparison.json",
                                                 "content_hash": _sha256(render_receipt_bytes)},
            "artifact_files": {
                "score": {"path": "artifacts/score.json", "content_hash": score_ref.content_hash},
                "bundle": {"path": "artifacts/bundle.tar", "content_hash": bundle_ref.content_hash},
                "snapshot": {"path": "artifacts/snapshot.json", "content_hash": snapshot_artifact.content_hash},
                "request": {"path": "artifacts/request.json", "content_hash": request_ref.content_hash},
                "finality": {"path": "artifacts/finality.json", "content_hash": finality_ref.content_hash},
                "model_evidence": {"path": "artifacts/model_evidence.json", "content_hash": model_evidence_ref.content_hash},
                "render": {"path": "artifacts/render.json", "content_hash": render_ref.content_hash},
                "release_manifest": {"path": "artifacts/release_manifest.json", "content_hash": _sha256(json.dumps(manifest, sort_keys=True).encode())},
                "score_spec": {"path": "artifacts/score_spec.json", "content_hash": _sha256(json.dumps(score_spec, sort_keys=True).encode())},
            },
            "expected_population": len(score_doc["expected_population"]),
        }
    finally:
        conn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output.parent / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    (receipt_dir / "score_comparison.json").write_bytes(score_receipt_bytes)
    (receipt_dir / "render_comparison.json").write_bytes(render_receipt_bytes)
    artifact_dir = args.output.parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for name, data in (("score", score_bytes), ("bundle", bundle_bytes), ("snapshot", snapshot_bytes),
                       ("request", request_bytes), ("finality", finality_bytes),
                       ("model_evidence", model_evidence_bytes), ("render", render_bytes)):
        suffix = ".tar" if name == "bundle" else ".json"
        (artifact_dir / f"{name}{suffix}").write_bytes(data)
    (artifact_dir / "release_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    (artifact_dir / "score_spec.json").write_text(json.dumps(score_spec, sort_keys=True))
    args.output.write_text(json.dumps(to_document(preview), indent=2, sort_keys=True) + "\n")
    args.output.with_name("source_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preview_input": str(args.output), "provenance": str(args.output.with_name("source_provenance.json")), "population": provenance["expected_population"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
