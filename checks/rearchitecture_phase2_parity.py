#!/usr/bin/env python3
"""D15 CLI: legacy-vs-snapshot score parity, over an existing ops catalog.

Reads two already-committed ``legacy_score`` jobs from ``--root`` (the same
``data/operations``-shaped directory ``engine/v2/ops/cli.py`` uses: a
``catalog.sqlite`` plus a content-addressed object store at its side) and
writes the ``comparison_receipt.v1.1`` document ``engine.v2.ops.parity``
builds under ``--artifact-root``, ready to be handed to
``checks/rearchitecture_phase2_evidence_build.py --score-receipt``.

Makes no provider calls and mutates nothing: strictly read-only over the
catalog and the store.
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
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.diagnosis.receipt import AGREE, ComparisonReceipt  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.parity import score_parity_receipt  # noqa: E402


def build(root: Path, *, legacy_job: str, snapshot_job: str) -> dict:
    """Open the catalog read-only-in-spirit, build and strictly re-decode the
    receipt document. ``code_hash``/``environment_hash`` are computed the
    SAME way ``rearchitecture_phase2_gate.py`` computes them, over THIS repo
    checkout (not the operations root) — "both Phase 2 inputs record the
    same code hash" (phase-2 guide §12.1). Re-decoding here (this module MAY
    import the diagnosis sink; ``engine.v2.ops.parity`` may not — see its
    module docstring) is the proof that ``score_parity_receipt``'s hand-built
    dict is a real ``ComparisonReceipt``, not merely shaped like one."""
    code_hash = source_hash(source_files(ROOT))
    environment_hash, _source = _environment_hash(ROOT)
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        store = ArtifactStore(root)
        doc = score_parity_receipt(conn, store, legacy_job_id=legacy_job,
                                   snapshot_job_id=snapshot_job, code_hash=code_hash,
                                   environment_hash=environment_hash)
    finally:
        conn.close()
    decode_document(ComparisonReceipt, doc)  # raises DocumentError if malformed
    return doc


def publish(doc: dict, artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, indent=2, sort_keys=True).encode()
    path = artifact_root / f"score_parity_{doc['receipt_id']}.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-job", required=True)
    parser.add_argument("--snapshot-job", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        doc = build(args.root, legacy_job=args.legacy_job, snapshot_job=args.snapshot_job)
    except OpsError as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc)}, indent=2))
        return 1
    ref = publish(doc, args.artifact_root)
    print(json.dumps({**ref, "verdict": doc["verdict"]}, indent=2))
    return 0 if doc["verdict"] == AGREE else 1


if __name__ == "__main__":
    raise SystemExit(main())
