"""Worker half of ``legacy_materialize`` (P2-6 §9.3, D13) — runs in the fixed
subprocess, never in the coordinator.

The worker reads only three things: the bound ``materialization_request.json``
the executor staged, the immutable object store, and the data catalog opened
through a read-only SQLite URI. It never sees the mutable legacy store.

One root per request hash, written once:

* absent -> ``materialize`` fills a private attempt-named sibling
  (``.<hex>.partial-<attempt_id>``), which ``materialize`` itself validates
  and locks down, and only then is it renamed onto ``<hex>`` in one atomic
  ``rename``. A crash leaves at most a partial directory nobody reads.
* present -> never rewritten: every file is re-hashed under the same mode and
  link checks the supervisor applies, and that manifest is reported.

The manifest document is deterministic (no attempt id, no reuse flag), so a
reused root yields the same artifact as the attempt that wrote it; the
coordinator compares the two before admitting it.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

from engine.v2.ops.snapshot_roots import (
    MANIFEST_SCHEMA_REF,
    hash_tree,
    manifest_document,
    materialization_root,
    partial_root,
)

__all__ = ["run_materialize"]


def _read_only_catalog(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _discard(path: Path) -> None:
    """Remove this attempt's own losing partial tree (read-only after lock-down)."""
    for directory, _dirs, _files in os.walk(path):
        os.chmod(directory, 0o755)
    shutil.rmtree(path, ignore_errors=True)


def _write_root(request, spec: dict, dest: Path, attempt_id: str):
    """Materialize into a private sibling, then rename. ``None`` if another
    attempt's rename won the race (the caller then reuses its root)."""
    from engine.v2.data.legacy_adapter import materialize
    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore

    base = Path(spec["base"])
    base.mkdir(parents=True, exist_ok=True)
    partial = partial_root(base, request.request_hash, attempt_id)
    store = ArtifactStore(spec["store_root"])
    conn = _read_only_catalog(spec["catalog_path"])
    try:
        files = materialize(Repository(conn, store), store, request, partial)
    finally:
        conn.close()
    try:
        os.rename(partial, dest)
    except OSError:
        if not dest.is_dir():
            raise
        _discard(partial)
        return None
    return files


def _px_absent_tickers(request, dest: Path) -> tuple[str, ...]:
    """Every ticker :func:`px_series_tickers` names that has no ``px_<T>.csv``
    under ``dest`` -- recomputed from the tree on disk rather than threaded
    through ``materialize``'s return value, so it is correct on BOTH the
    fresh-write and the reused-root path (a reused root never calls
    ``materialize``/``materialize_price_series`` again, task brief
    2026-09-15's ``px_absent_tickers`` requirement). Mirrors the same
    snapshot-has-no-price_history-table guard ``legacy_adapter.materialize``
    uses before calling ``materialize_price_series`` at all."""
    from engine.v2.data import legacy_materialization

    if legacy_materialization.PRICE_HISTORY_TABLE_NAME not in request.snapshot_ref.table_versions:
        return ()
    px_tickers = legacy_materialization.px_series_tickers(request)
    return tuple(sorted(t for t in px_tickers
                        if not legacy_materialization.px_csv_path(dest, t).is_file()))


def run_materialize(parameters, root: Path, envelope: dict) -> dict:
    from engine.v2.contracts import LegacyMaterializationRequest
    from engine.v2.data.documents import decode_document

    spec = envelope["materialization"]
    request = decode_document(LegacyMaterializationRequest, json.loads(
        (root / "materialization_request.json").read_text()))
    dest = materialization_root(spec["base"], request.request_hash)
    files = None
    if not (dest.exists() or dest.is_symlink()):
        files = _write_root(request, spec, dest, envelope["attempt_id"])
    reused = files is None
    if reused:
        files = hash_tree(dest)
    document = manifest_document(request, files)
    (root / "materialization_manifest.json").write_text(json.dumps(document, sort_keys=True))
    px_absent_tickers = _px_absent_tickers(request, dest)
    return {"outputs": [{"name": "materialization_manifest", "path": "materialization_manifest.json",
                         "schema": MANIFEST_SCHEMA_REF}],
            "completed_ids": list(parameters["expected_ids"]),
            "no_work": not parameters["expected_ids"], "reused": reused,
            "file_count": len(files),
            "px_absent_tickers": list(px_absent_tickers),
            "px_absent_count": len(px_absent_tickers)}
