#!/usr/bin/env python3
"""Offline projection coordinator — rearchitecture phase-3 guide §5.4 (P3-1b).

Reads an already-saved ``score.json`` and a flat per-ticker render bundle
directory, resolves their events through the pinned Phase 2 repository, and
calls :func:`engine.v2.serving.projections.build_candidate` to publish and
index one release. Prints ``{"release_id": ..., "findings": {...}}`` (or the
refusal ``Problem``) as one JSON document to stdout.

    python3 tools/v2_dashboard_project.py \\
        --preview-input preview_input.json --score-json score.json \\
        --bundle-dir bundle/tickers --snapshot-id snap_... \\
        --catalog catalog.sqlite --store-root store \\
        --serving-root serving --requested-as-of 2026-01-14 \\
        --resolved-as-of 2026-01-14

No scoring, no provider/network calls, no legacy ``engine.*`` import — every
input is already on disk. ``tools/`` composes across layers (guide §2), so
this is the one place allowed to import both ``engine.v2.ops.bootstrap``
(opening the Phase 2 catalog) and ``engine.v2.serving`` in one process; the
serving package itself never imports ops.

**Bundle-directory shape, a deliberate simplification.** The real
``dashboard/render.py`` per-ticker file (``data/tickers/<ticker>.json``) is a
nested, event-grouped evidence payload, not the flat ``list[dict]`` the
bridge consumes. Parsing that real shape is real-bundle integration —
explicitly deferred, like the rest of "real parity", to P3-4. This tool reads
the bridge's own native shape instead: one ``<ticker>.json`` file per ticker,
each a JSON array of ``compact_row``-shaped dicts — exactly what
``tests/test_v2_serving_bridge.py``'s synthetic fixtures already use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import PreviewInput, PreviewRelease, Problem  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactStore,
    SystemClock,
    from_document,
    to_document,
)
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.serving.projections import build_candidate, connect  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview-input", required=True, type=Path,
                        help="PreviewInput document (JSON)")
    parser.add_argument("--score-json", required=True, type=Path,
                        help="saved score.json (expected_population/rows/ladder)")
    parser.add_argument("--bundle-dir", required=True, type=Path,
                        help="directory of <ticker>.json render-bundle rows")
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


def _load_score_doc(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_bundle(bundle_dir: Path) -> dict[str, list[dict]]:
    bundle: dict[str, list[dict]] = {}
    for path in sorted(bundle_dir.glob("*.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array of rendered rows")
        bundle[path.stem] = rows
    return bundle


def _result_document(result: PreviewRelease | Problem, conn) -> dict:
    if isinstance(result, Problem):
        return {"ok": False, "problem": to_document(result)}
    row = conn.execute("SELECT findings_json FROM serving_release WHERE release_id = ?",
                       (result.release_id,)).fetchone()
    findings = json.loads(row["findings_json"]) if row is not None else None
    return {"ok": True, "release_id": result.release_id, "release": to_document(result),
            "findings": findings}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    preview_input = from_document(PreviewInput, json.loads(args.preview_input.read_text()))
    score_doc = _load_score_doc(args.score_json)
    bundle_rows_by_ticker = _load_bundle(args.bundle_dir)

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
