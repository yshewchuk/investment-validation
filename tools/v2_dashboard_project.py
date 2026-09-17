#!/usr/bin/env python3
"""Offline projection coordinator — rearchitecture phase-3 guide §5.4 (P3-1b/P3-4).

Reads an already-saved ``score.json`` and a rendered bundle, resolves their
events through the pinned Phase 2 repository, and calls
:func:`engine.v2.serving.projections.build_candidate` to publish and index
one release. Prints ``{"release_id": ..., "findings": {...}}`` (or the
refusal ``Problem``) as one JSON document to stdout.

    python3 tools/v2_dashboard_project.py \\
        --preview-input preview_input.json --score-json score.json \\
        --bundle-dir bundle --snapshot-id snap_... \\
        --catalog catalog.sqlite --store-root store \\
        --serving-root serving --requested-as-of 2026-01-14 \\
        --resolved-as-of 2026-01-14

No scoring, no provider/network calls, no legacy ``engine.*`` import — every
input is already on disk. ``tools/`` composes across layers (guide §2), so
this is the one place allowed to import both ``engine.v2.ops.bootstrap``
(opening the Phase 2 catalog) and ``engine.v2.serving`` in one process; the
serving package itself never imports ops.

**Bundle format (P3-4).** ``--bundle-format legacy`` (the default) reads the
real ``engine/dashboard/render.py`` ``render_bundle`` output tree —
``data/board.json``/``data/tickers/<ticker>.json`` (or their ``.js``
wrappers) — through :func:`engine.v2.serving.legacy_bundle.load_legacy_bundle`.
Every file that loader reads is hashed into a manifest; this tool folds
``content_hash(bundle_manifest)`` into the ``PreviewInput`` it passes to
``build_candidate`` as ``bundle_manifest_ref``, overriding whatever value the
``--preview-input`` document carried — so the release's identity binds to the
bundle's *actual* bytes, not a caller-declared reference. ``--bundle-format
flat`` keeps the pre-P3-4 simplification (one ``<ticker>.json`` file per
ticker, each a JSON array of ``compact_row``-shaped dicts) for tests that
predate the real adapter; it leaves ``--preview-input``'s own
``bundle_manifest_ref`` untouched.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import PreviewInput, PreviewRelease, Problem  # noqa: E402
from engine.v2.diagnosis.receipt import ComparisonReceipt  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactStore,
    SystemClock,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from engine.v2.serving.legacy_bundle import load_legacy_bundle, load_score_document  # noqa: E402
from engine.v2.serving.projections import build_candidate, connect  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview-input", required=True, type=Path,
                        help="PreviewInput document (JSON)")
    parser.add_argument("--source-provenance", type=Path,
                        help="verifier-produced source_provenance.json paired with preview input")
    parser.add_argument("--score-json", required=True, type=Path,
                        help="saved score.json (expected_population/rows/ladder)")
    parser.add_argument("--bundle-dir", required=True, type=Path,
                        help="'legacy' format: the rendered bundle ROOT (the directory "
                        "holding data/board.json, data/tickers/<ticker>.json). "
                        "'flat' format: a directory of <ticker>.json render-bundle-row "
                        "arrays (the pre-P3-4 simplification, kept for existing tests).")
    parser.add_argument("--bundle-format", choices=("flat", "legacy"), default="legacy",
                        help="'legacy' (default): read the real dashboard/render.py bundle "
                        "layout via engine.v2.serving.legacy_bundle.load_legacy_bundle, "
                        "byte-hashed into the release's bundle_manifest_ref. "
                        "'flat': the simplified one-array-per-ticker shape; kept only for "
                        "tests that predate the real adapter.")
    parser.add_argument("--snapshot-id", required=True,
                        help="the pinned Phase 2 snapshot id to resolve events against")
    parser.add_argument("--catalog", required=True, type=Path,
                        help="the Phase 2 data catalog sqlite file")
    parser.add_argument("--store-root", required=True, type=Path,
                        help="the Phase 2 ArtifactStore root (read-only here)")
    parser.add_argument("--serving-root", required=True, type=Path,
                        help="output directory: serving.sqlite plus this release's own objects/")
    parser.add_argument("--requested-as-of", required=True)
    parser.add_argument("--resolved-as-of", required=True)
    return parser.parse_args(argv)


def _load_flat_bundle(bundle_dir: Path) -> dict[str, list[dict]]:
    bundle: dict[str, list[dict]] = {}
    for path in sorted(bundle_dir.glob("*.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array of rendered rows")
        bundle[path.stem] = rows
    return bundle


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _verify_source_provenance(preview: PreviewInput, path: Path, *, score_path: Path,
                              bundle_manifest_ref: str) -> None:
    """Refuse detached score/bundle inputs at the real coordinator boundary.

    The verifier has already used ArtifactStore.read_verified against the
    Phase 2 store. This second check binds the files delivered to this command
    to that retained proof, rather than allowing a caller to swap either one
    after the verifier ran.
    """
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema_version") != "phase3_source_provenance.v1.0":
        raise ValueError("source provenance has an unsupported schema")
    if doc.get("release_id") != preview.source_release_id or \
            doc.get("release_manifest_hash") != preview.source_release_manifest_ref:
        raise ValueError("source provenance does not bind this source release")
    refs = {name: doc.get(name) for name in ("score_artifact", "snapshot_artifact",
            "materialization_request_artifact", "finality_artifact", "model_evidence_artifact")}
    if any(not isinstance(ref, dict) for ref in refs.values()):
        raise ValueError("source provenance has missing retained artifact refs")
    if refs["score_artifact"].get("content_hash") != preview.score_batch_ref or \
            refs["finality_artifact"].get("content_hash") != preview.finality_ref or \
            refs["model_evidence_artifact"].get("content_hash") != preview.model_evidence_ref:
        raise ValueError("source provenance and preview artifact bindings disagree")
    if _sha256(score_path.read_bytes()) != refs["score_artifact"].get("content_hash"):
        raise ValueError("score json is not the verified score artifact")
    if doc.get("bundle_manifest_ref") != bundle_manifest_ref:
        raise ValueError("bundle directory is not the verified delivered bundle")
    for name, expected_kind, expected_ref in (
            ("score_comparison_receipt", "score_record_parity", preview.score_comparison_receipt_ref),
            ("render_comparison_receipt", "render_bundle_parity", preview.render_comparison_receipt_ref)):
        ref = doc.get(name)
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise ValueError(f"source provenance is missing {name}")
        receipt_path = (path.parent / ref["path"]).resolve()
        try:
            receipt_path.relative_to(path.parent.resolve())
        except ValueError:
            raise ValueError(f"{name} escapes source provenance directory")
        receipt_bytes = receipt_path.read_bytes()
        receipt = from_document(ComparisonReceipt, json.loads(receipt_bytes))
        if _sha256(receipt_bytes) != expected_ref or ref.get("content_hash") != expected_ref \
                or receipt.comparison_kind != expected_kind:
            raise ValueError(f"{name} does not bind the verified preview input")


def _result_document(result: PreviewRelease | Problem, conn) -> dict:
    if isinstance(result, Problem):
        return {"ok": False, "problem": to_document(result)}
    row = conn.execute("SELECT findings_json FROM serving_release WHERE release_id = ?",
                       (result.release_id,)).fetchone()
    findings = json.loads(row["findings_json"]) if row is not None else None
    # guide §5.4/§8 P3-1 step 3 (P3-1c): the operator entry point for
    # binding this candidate to the existing fenced ops publisher is "a
    # publication-effect input" (engine.v2.ops.effects_graph.
    # publication_effect now accepts an optional named "projection_binding.
    # json" input binding, the same mechanism bundle.tar/finality.json
    # already use) -- not a new coordinator stage here. This tool's own
    # contribution stays the smaller half: emit the binding document
    # alongside the release it already prints, so an operator/submission
    # script can register these bytes as an ops artifact and bind them into
    # the publication job exactly the way it already binds the render
    # bundle. Never published/staged here -- this tool stays offline.
    binding = projections.projection_binding(conn, result.release_id)
    return {"ok": True, "release_id": result.release_id, "release": to_document(result),
            "findings": findings, "projection_binding": binding}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    preview_input = from_document(PreviewInput, json.loads(args.preview_input.read_text()))
    score_doc = load_score_document(args.score_json)

    bundle_manifest_ref = preview_input.bundle_manifest_ref
    if args.bundle_format == "flat":
        bundle_rows_by_ticker = _load_flat_bundle(args.bundle_dir)
    else:
        bundle_rows_by_ticker, bundle_manifest = load_legacy_bundle(args.bundle_dir)
        # The release's identity binds to the bundle's actual bytes, not to
        # whatever bundle_manifest_ref the caller's PreviewInput happened to
        # declare — a changed byte anywhere in the bundle must change the
        # release id (§5.3 point 8's idempotency/change-detection test).
        bundle_manifest_ref = content_hash(bundle_manifest)
        preview_input = dataclasses.replace(preview_input, bundle_manifest_ref=bundle_manifest_ref)

    # Synthetic unit fixtures use symbolic refs. Real Phase 2 releases always
    # carry a content hash, and therefore must supply the verifier proof.
    if preview_input.source_release_manifest_ref.startswith("sha256:"):
        if args.source_provenance is None:
            raise ValueError("real source releases require --source-provenance")
        _verify_source_provenance(preview_input, args.source_provenance,
                                  score_path=args.score_json, bundle_manifest_ref=bundle_manifest_ref)

    clock = SystemClock()
    catalog_conn = open_catalog(args.catalog, clock=clock)
    phase2_store = ArtifactStore(args.store_root)
    repository = Repository(catalog_conn, phase2_store)
    snapshot_ref = repository.resolve(args.snapshot_id)

    args.serving_root.mkdir(parents=True, exist_ok=True)
    serving_store = ArtifactStore(args.serving_root / "objects")
    serving_conn = connect(str(args.serving_root / "serving.sqlite"), clock=clock)

    result = build_candidate(
        preview_input, score_doc, bundle_rows_by_ticker,
        repository=repository, snapshot_ref=snapshot_ref, store=serving_store, conn=serving_conn,
        requested_as_of=args.requested_as_of, resolved_as_of=args.resolved_as_of, clock=clock)

    print(json.dumps(_result_document(result, serving_conn), indent=2, sort_keys=True))
    serving_conn.close()
    catalog_conn.close()
    return 0 if isinstance(result, PreviewRelease) else 1


if __name__ == "__main__":
    raise SystemExit(main())
