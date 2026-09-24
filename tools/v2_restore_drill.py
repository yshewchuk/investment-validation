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
  file-set- and byte-for-byte against a generation already exported from the
  live catalog before the backup was taken, IN BOTH DIRECTIONS (this is the
  "report" side of the drill; the "ledger" side is the restored ``decisions``
  row count). A matching decision count never excuses absent, extra or
  differing export files.

Original-export baseline policy (review C6)::

* a missing, symlinked or otherwise unusable ``--original-export`` is
  REFUSED (exit 2) before anything is restored;
* symlinks and other non-regular entries inside either tree are reported
  as ``files_untrusted`` and never followed or read as evidence;
* an empty (or otherwise file-less) baseline directory is affirmative
  zero-decision evidence ONLY when the operator passes
  ``--expected-decisions-count 0``, the parent export root carries the
  ``CURRENT`` pointer ``export_generation`` wrote naming this generation,
  the baseline directory is itself the ``<root>/<generation>`` that pointer
  names, and the restored side is genuinely empty and zero-decision too; an
  arbitrary empty directory never passes;
* ``verdict`` is PASS only when every shared file matched bytes, no file
  is missing/extra/untrusted, no baseline problem stands, and -- when
  given -- the expected decision count matches.

Usage::

    python3 tools/v2_restore_drill.py \\
        --backup /path/to/backup --restore-root /path/to/restored \\
        --score-request request.json --native-inputs native_inputs.json \\
        --expected-score-hash sha256:... \\
        --original-export /path/to/live-export-root --generation g1 \\
        --expected-decisions-count 3 \\
        [--artifact-root evidence/]

Exits 0 and prints the receipt on PASS; exits 1 and prints the receipt (to
stderr's twin on stdout) on FAIL; exits 2 with a refusal line when the
backup cannot be restored or the original-export baseline is unusable.
Never mutates the backup or the original export directory.
"""
from __future__ import annotations

import argparse
import json
import os
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
    """The backup could not be restored at all, or the original-export
    baseline is unusable (missing, symlinked, not a directory, unreadable)
    -- either way, not a reconciliation mismatch."""


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


def _classify_tree(root: Path) -> dict[str, str]:
    """Relative posix path -> entry kind ('file', 'dir', 'symlink', 'other').

    Gathered with lstat semantics: NO symlink under ``root`` is ever
    followed, and a symlink never contributes file bytes as evidence.
    """
    kinds: dict[str, str] = {}
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        base, prefix = stack.pop()
        try:
            with os.scandir(base) as entries:
                for entry in entries:
                    relative = prefix + entry.name
                    if entry.is_symlink():
                        kinds[relative] = "symlink"
                    elif entry.is_dir(follow_symlinks=False):
                        kinds[relative] = "dir"
                        stack.append((Path(entry.path), relative + "/"))
                    elif entry.is_file(follow_symlinks=False):
                        kinds[relative] = "file"
                    else:
                        kinds[relative] = "other"
        except OSError as exc:
            raise RestoreDrillError(f"cannot read export tree {base}: {exc}") from exc
    return kinds


def _unattested_file_less_baseline(original_export_dir: Path, generation: str,
                                   expected_decisions_count: int | None) -> str | None:
    """Explicit policy for a file-less baseline directory (documented in the
    module docstring): a legitimate zero-decision ``export_generation`` run
    leaves an EMPTY generation directory plus a ``CURRENT`` pointer naming
    it, so accept one only on affirmative expected-zero evidence --
    ``expected_decisions_count == 0`` AND a ``CURRENT`` in the parent export
    root that is a regular file containing exactly ``generation + "\\n"`` AND
    the baseline directory itself being the ``<root>/<generation>`` that
    pointer names -- an empty sibling of a genuine export root is no
    attestation. The name is compared lexically; no symlink is followed or
    resolved. Returns the refusal reason, or None when the baseline IS
    attested."""
    if expected_decisions_count != 0:
        return ("original export baseline holds no files and no affirmative "
                "expected_decisions_count of 0 was given; an empty directory is "
                "not proof of a legitimate zero-decision export")
    pointer = original_export_dir.parent / "CURRENT"
    if pointer.is_symlink() or not pointer.is_file():
        return ("original export baseline holds no files and its parent export "
                "root carries no CURRENT pointer file; this is not a generation "
                "export_generation produced")
    try:
        named = pointer.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"original export baseline's CURRENT pointer is unreadable: {exc}"
    expected_pointer = generation + "\n"
    if named != expected_pointer:
        return (f"original export baseline holds no files and its CURRENT pointer "
                f"names {named!r}, not the expected {expected_pointer!r}")
    if original_export_dir.name != generation:
        return (f"original export baseline holds no files and {original_export_dir} "
                f"is not the directory CURRENT names ({pointer.parent / generation!r}); "
                "an empty sibling of a genuine export root is not attested evidence")
    return None


def _reconcile_ledger(restored_conn, export_root: Path, generation: str,
                      original_export_dir: Path,
                      expected_decisions_count: int | None) -> dict:
    """The restored catalog's own export, reconciled against a generation
    already exported from the LIVE catalog before the backup: complete
    relative file sets in BOTH directions (missing and extra both fail),
    every shared regular file byte-compared, path kinds compared without
    ever following a symlink. A matching decision count never excuses an
    absent, extra, untrusted or byte-differing file, and a file-less
    baseline must pass the affirmative-zero attestation above."""
    restored_export_dir = export_generation(restored_conn, export_root, generation=generation)
    original = _classify_tree(original_export_dir)
    restored = _classify_tree(restored_export_dir)

    shared = set(original) & set(restored)
    files_missing_in_restored = sorted(set(original) - set(restored))
    files_extra_in_restored = sorted(set(restored) - set(original))
    files_type_mismatched = sorted(p for p in shared if original[p] != restored[p])
    byte_compared = sorted(p for p in shared if original[p] == restored[p] == "file")
    files_mismatched = []
    for relative in byte_compared:
        if ((original_export_dir / relative).read_bytes()
                != (restored_export_dir / relative).read_bytes()):
            files_mismatched.append(relative)
    entries_untrusted = (
        [f"original/{p} ({kind})" for p, kind in sorted(original.items())
         if kind not in ("file", "dir")]
        + [f"restored/{p} ({kind})" for p, kind in sorted(restored.items())
           if kind not in ("file", "dir")])
    baseline_problem = None
    if "file" not in original.values():
        baseline_problem = _unattested_file_less_baseline(original_export_dir, generation,
                                                          expected_decisions_count)
    count = len(decision_rows(restored_conn))
    files_ok = not (files_missing_in_restored or files_extra_in_restored
                    or files_type_mismatched or files_mismatched or entries_untrusted
                    or baseline_problem)
    return {
        "generation": generation,
        "original_export_dir": str(original_export_dir),
        "restored_export_dir": str(restored_export_dir),
        "files_compared": len(byte_compared),
        "files_mismatched": files_mismatched,
        "files_missing_in_restored": files_missing_in_restored,
        "files_extra_in_restored": files_extra_in_restored,
        "files_type_mismatched": files_type_mismatched,
        "entries_untrusted": entries_untrusted,
        "baseline_problem": baseline_problem,
        "decisions_count": count,
        "expected_decisions_count": expected_decisions_count,
        "match": files_ok and (expected_decisions_count is None
                               or count == expected_decisions_count),
    }


def run_drill(*, backup_dir: Path | str, restore_root: Path | str,
             score_request_name: str, native_inputs_name: str, expected_score_hash: str,
             original_export_dir: Path | str, generation: str,
             expected_decisions_count: int | None = None,
             artifact_root: Path | str | None = None) -> dict:
    backup_dir, restore_root = Path(backup_dir), Path(restore_root)
    original_export_dir = Path(original_export_dir)
    if original_export_dir.is_symlink():
        raise RestoreDrillError(f"original export directory {original_export_dir} is a "
                                "symlink; a symlinked baseline is never trusted evidence")
    if not original_export_dir.is_dir():
        raise RestoreDrillError(f"original export directory {original_export_dir} does not "
                                "exist or is not a directory; the drill would reconcile zero "
                                "baseline files, which can never yield PASS")
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

        ledger = _reconcile_ledger(conn, restored_root / "export", generation,
                                   original_export_dir, expected_decisions_count)
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
