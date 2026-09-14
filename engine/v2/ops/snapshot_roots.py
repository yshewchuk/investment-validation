"""Filesystem half of snapshot-backed legacy stages (P2-6 §9.3, D13).

A materialization root is a derived, read-only directory tree that
``engine.v2.data.legacy_adapter.materialize`` wrote for exactly one
``LegacyMaterializationRequest``. Nothing here trusts a path's presence: every
check walks the real tree with ``lstat`` (never following a link) and compares
it against the declared ``{relative_path: content_hash}`` manifest.

Location: roots live beside the operations root, never beneath it
(``<ops_root>.materializations/<request hex>``). ``materialize`` refuses any
destination under the artifact store's own root, and the supervisor's store
root IS the operations root, so ``<ops_root>/materializations`` is refused by
the data layer by design.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash

__all__ = ["MANIFEST_SCHEMA_REF", "default_materialization_base", "hash_tree",
           "manifest_document", "materialization_root", "partial_root", "stat_fingerprint",
           "verify_root"]

MANIFEST_SCHEMA_REF = "legacy_materialization_manifest.v1.0"
_FILE_MODE = 0o444
_DIR_MODE = 0o555


def default_materialization_base(ops_root) -> Path:
    root = Path(ops_root).resolve()
    return root.parent / (root.name + ".materializations")


def _hex(request_hash: str) -> str:
    digest = str(request_hash).removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise fail("INPUT_CHANGED", "materialization request hash is malformed")
    return digest


def materialization_root(base, request_hash: str) -> Path:
    return Path(base) / _hex(request_hash)


def partial_root(base, request_hash: str, attempt_id: str) -> Path:
    """A private, attempt-named sibling a worker fills before its atomic rename."""
    return Path(base) / f".{_hex(request_hash)}.partial-{attempt_id}"


def manifest_document(request, files: dict) -> dict:
    """Deterministic: no attempt id, no reuse flag — identical trees share one artifact."""
    return {"schema_version": MANIFEST_SCHEMA_REF, "request_hash": request.request_hash,
            "snapshot_id": request.snapshot_ref.snapshot_id,
            "snapshot_manifest_hash": request.snapshot_ref.manifest_hash,
            "files": {path: files[path] for path in sorted(files)}}


def _walk(root: Path):
    """Yield ``(relative_path, lstat)`` for every file, refusing anything but
    0555 directories and singly-linked 0444 regular files."""
    try:
        top = os.lstat(root)
    except OSError:
        raise fail("INPUT_CHANGED", "materialization root is missing") from None
    if not stat.S_ISDIR(top.st_mode) or stat.S_IMODE(top.st_mode) != _DIR_MODE:
        raise fail("INPUT_CHANGED", "materialization root is indirect or writable")
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                relative = Path(entry.path).relative_to(root).as_posix()
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode) and mode == _DIR_MODE:
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode) and mode == _FILE_MODE and info.st_nlink == 1:
                    yield relative, info
                else:
                    raise fail("INPUT_CHANGED", "materialization root holds an indirect, linked "
                               "or writable entry", details={"path": relative})


def stat_fingerprint(root) -> dict:
    """Cheap identity of every file: inode, size, mtime. No hashing."""
    return {rel: [info.st_ino, info.st_size, info.st_mtime_ns]
            for rel, info in sorted(_walk(Path(root)))}


def hash_tree(root) -> dict:
    """``{relative_path: content_hash}`` of a locked-down tree (modes enforced)."""
    root = Path(root)
    return {rel: file_hash(root / rel) for rel, _ in sorted(_walk(root))}


def verify_root(root, files: dict) -> dict:
    """Refuse ``INPUT_CHANGED`` unless ``root`` holds exactly ``files``, byte for byte.

    Returns the tree's :func:`stat_fingerprint`, taken around the hashing pass,
    so a later cheap re-check can detect a replaced or rewritten file.
    """
    root = Path(root)
    fingerprint = stat_fingerprint(root)
    if set(fingerprint) != set(files):
        raise fail("INPUT_CHANGED", "materialization root differs from its manifest",
                   details={"missing": sorted(set(files) - set(fingerprint))[:5],
                            "undeclared": sorted(set(fingerprint) - set(files))[:5]})
    for rel in sorted(files):
        if file_hash(root / rel) != files[rel]:
            raise fail("INPUT_CHANGED", "materialization file changed", details={"path": rel})
    if stat_fingerprint(root) != fingerprint:
        raise fail("INPUT_CHANGED", "materialization root changed while it was verified")
    return fingerprint
