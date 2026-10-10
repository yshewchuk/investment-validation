"""Reclaim read-once attempt staging and orphan partial materialization roots.

Contract: engine/v2/ops/ARCHITECTURE.md, "Attempt staging reclaim" (#609, #467).
Only terminal attempts whose process is settled are eligible; running, queued
and recovering attempts are never touched. Eligibility is the entry's own
existence, so a crash mid-removal is finished by the next pass.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from engine.v2.ops.snapshot_roots import default_materialization_base

__all__ = ["READ_ONCE", "reclaim"]

#: Subtrees of ``attempts/<id>/staging`` the worker reads only during the attempt.
READ_ONCE = ("legacy",)
_STATES = ("succeeded", "failed", "cancelled")
_PROCESSES = ("unlaunched", "exited", "verified_dead")
_PARTIAL = re.compile(r"^\.[0-9a-f]{64}\.partial-(?P<attempt>.+)$")


def _eligible(conn):
    sql = ("SELECT attempt_id FROM attempts WHERE state IN (?,?,?) "
           "AND process_state IN (?,?,?) ORDER BY created_at")
    return [row[0] for row in conn.execute(sql, _STATES + _PROCESSES)]


def _size(path):
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _remove(path):
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    for dirpath, _dirs, _files in os.walk(path):
        os.chmod(dirpath, 0o700)
    shutil.rmtree(path)


def _entry(kind, path, attempt_id, apply, out):
    if not os.path.lexists(path):
        return
    record = {"kind": kind, "attempt_id": attempt_id, "path": str(path),
              "bytes": _size(path), "removed": False}
    if apply:
        try:
            _remove(path)
            record["removed"] = True
        except OSError as exc:
            record["error"] = type(exc).__name__
    out.append(record)


def reclaim(conn, root, *, apply=False, limit=None, materialization_base=None):
    """List (default) or remove every eligible read-once entry.

    ``limit`` stops after that many successful removals; a failed removal is
    reported in its record (``error``) and never raises or counts toward it.
    """
    root, out = Path(root), []

    def full():
        return limit is not None and sum(r["removed"] for r in out) >= limit

    attempts = _eligible(conn)
    for attempt_id in attempts:
        if full():
            return out
        if Path(attempt_id).name != attempt_id:
            continue
        for name in READ_ONCE:
            _entry("staging", root / "attempts" / attempt_id / "staging" / name,
                   attempt_id, apply, out)
    base = (Path(materialization_base) if materialization_base
            else default_materialization_base(root))
    if base.is_dir():
        eligible = set(attempts)
        for entry in sorted(base.iterdir()):
            match = _PARTIAL.match(entry.name)
            if match and match["attempt"] in eligible:
                if full():
                    break
                _entry("partial_materialization", entry, match["attempt"], apply, out)
    return out