#!/usr/bin/env python3
"""P6-5 export-restore-drill: restore a deployment from a LOCAL backup,
replay one original score offline, and reconcile ledger/report against it.

UD-3 (decided): git push / private-mirror sync stays an operator action.
This drill never pushes anything and never touches a remote -- it assumes a
local backup (``engine.v2.ops.backup.run_backup`` output: a directory holding
``<key>.manifest.json``, ``<key>.sqlite`` and ``artifacts/``) already exists,
and only reads it.

Three real production entrypoints, composed, nothing re-implemented:

* ``engine.v2.ops.backup.restore_backup`` -- restore into a brand-new,
  isolated root (refuses to overwrite an existing one) and verify every
  database/artifact byte against the manifest.
* ``engine.v2.ops.cli.rescore_command`` -- the SAME read-only, no-fit,
  no-provider-pull entrypoint ``ops rescore`` exposes, replaying one
  ``ScoreRequest``/``NativeScoreInputs`` pair (already-captured data,
  restored from the backup's ``artifacts/``) into a ``ScoreRecord``.
* ``engine.v2.ledger.export.export_generation`` -- the same projection the
  nightly ``ledger_export`` effect runs, over the RESTORED catalog, compared
  byte-for-byte against a generation already exported from the live catalog
  before the backup was taken (this is the "report" side of the drill; the
  "ledger" side is the restored ``decisions`` row count).

Usage::

    python3 tools/v2_restore_drill.py \\
        --backup /path/to/backup --restore-root /path/to/restored \\
        --score-request request.json --native-inputs native_inputs.json \\
        --expected-score-hash sha256:... \\
        --original-export /path/to/live-export-root --generation g1 \\
        --expected-decisions-count 3 \\
        [--artifact-root evidence/]

Exits 0 and prints the receipt on PASS; exits 1 and prints the receipt (to
stderr's twin on stdout) on FAIL. Never mutates the backup or the original
export directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import SystemClock, content_hash, to_document  # noqa: E402
from engine.v2.ledger.decisions import rows as decision_rows  # noqa: E402
from engine.v2.ledger.export import export_generation  # noqa: E402
from engine.v2.ops.backup import restore_backup  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.cli import rescore_command  # noqa: E402

__all__ = ["RestoreDrillError", "run_drill", "main"]

SCHEMA_VERSION = "restore_drill_receipt.v1.0"


class RestoreDrillError(RuntimeError):
    """The backup could not be restored at all (not a reconciliation mismatch)."""


def _replay_score(restored_root: Path, score_request_name: str, native_inputs_name: str) -> dict:
    """Replay one score entirely from restored bytes: no provider pulls, no fitting.

    Copies the two restored artifact files into a private scratch directory
    under the names ``rescore_command`` expects, so this never depends on
    the backup's original artifact naming matching any particular suffix.
    """
    request_bytes = (restored_root / "artifacts" / score_request_name).read_bytes()
    native_bytes = (restored_root / "artifacts" / native_inputs_name).read_bytes()
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        request_path = scratch_dir / "request.json"
        native_path = scratch_dir / "native_inputs.json"
        request_path.write_bytes(request_bytes)
        native_path.write_bytes(native_bytes)
        record = rescore_command(SimpleNamespace(request=request_path, native_inputs=native_path))
    document = to_document(record)
    return {"score_id": getattr(record, "score_id", None), "content_hash": content_hash(document)}


def _reconcile_ledger(restored_conn, export_root: Path, generation: str,
                      original_export_dir: Path) -> dict:
    """The restored catalog's own export, diffed byte-for-byte against a
    generation already exported from the LIVE catalog before the backup."""
    restored_export_dir = export_generation(restored_conn, export_root, generation=generation)
    original_files = sorted(p for p in original_export_dir.rglob("*") if p.is_file())
    mismatched = []
    for path in original_files:
        relative = path.relative_to(original_export_dir)
        twin = restored_export_dir / relative
        if not twin.is_file() or twin.read_bytes() != path.read_bytes():
            mismatched.append(str(relative))
    count = len(decision_rows(restored_conn))
    return {
        "generation": generation,
        "files_compared": len(original_files),
        "files_mismatched": mismatched,
        "decisions_count": count,
    }


def run_drill(*, backup_dir: Path | str, restore_root: Path | str,
             score_request_name: str, native_inputs_name: str, expected_score_hash: str,
             original_export_dir: Path | str, generation: str,
             expected_decisions_count: int | None = None,
             artifact_root: Path | str | None = None) -> dict:
    backup_dir, restore_root = Path(backup_dir), Path(restore_root)
    original_export_dir = Path(original_export_dir)
    try:
        restored_root = restore_backup(backup_dir, restore_root)
    except Exception as exc:  # restore_backup raises engine.v2.ops.errors.OpsError subclasses
        raise RestoreDrillError(f"restore failed: {exc}") from exc

    clock = SystemClock()
    conn = open_catalog(restored_root / "ops.sqlite", clock=clock)
    try:
        score_replay = _replay_score(restored_root, score_request_name, native_inputs_name)
        score_replay["expected_hash"] = expected_score_hash
        score_replay["match"] = score_replay["content_hash"] == expected_score_hash

        ledger = _reconcile_ledger(conn, restored_root / "export", generation, original_export_dir)
        ledger["expected_decisions_count"] = expected_decisions_count
        ledger["match"] = not ledger["files_mismatched"] and (
            expected_decisions_count is None or ledger["decisions_count"] == expected_decisions_count)
    finally:
        conn.close()

    verdict = "PASS" if score_replay["match"] and ledger["match"] else "FAIL"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "restored_root": str(restored_root),
        "score_replay": score_replay,
        "ledger_reconciliation": ledger,
        "verdict": verdict,
    }
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "restore_drill_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path, dest="backup_dir")
    parser.add_argument("--restore-root", required=True, type=Path)
    parser.add_argument("--score-request", required=True, dest="score_request_name")
    parser.add_argument("--native-inputs", required=True, dest="native_inputs_name")
    parser.add_argument("--expected-score-hash", required=True)
    parser.add_argument("--original-export", required=True, type=Path, dest="original_export_dir")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--expected-decisions-count", type=int, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = run_drill(
            backup_dir=args.backup_dir, restore_root=args.restore_root,
            score_request_name=args.score_request_name, native_inputs_name=args.native_inputs_name,
            expected_score_hash=args.expected_score_hash,
            original_export_dir=args.original_export_dir, generation=args.generation,
            expected_decisions_count=args.expected_decisions_count, artifact_root=args.artifact_root)
    except RestoreDrillError as exc:
        print(f"restore-drill: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
