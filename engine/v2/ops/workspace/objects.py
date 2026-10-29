"""Admit exactly the pinned snapshot's closure objects into the workspace store."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import time

from engine.v2.data.errors import DataError
from engine.v2.foundation.artifacts import ensure_directory, fsync_directory
from engine.v2.ops.errors import OpsError, fail

_MAX_OBJECTS = 1_000_000
_MAX_BYTES = 16 << 30
_PROGRESS_SECONDS = 30
_CHUNK = 1 << 20
_HASH = re.compile(r"sha256:([0-9a-f]{64})\Z")
_NAMED = (
    "SELECT DISTINCT substr(o.object_id, 1, 201), substr(o.content_hash, 1, 201), o.byte_size, substr(o.storage_key, 1, 201) FROM data_objects o "
    "JOIN data_fragments f ON f.object_id = o.object_id "
    "JOIN data_version_fragments vf ON vf.fragment_id = f.fragment_id "
    "JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id "
    "WHERE st.snapshot_id = ? ORDER BY 1"
)


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace object admission refused",
                details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


def _present(path, size: int, digest: str, object_id: str) -> bool:
    """True if a complete, hash-verified destination exists; a bad one is never repaired."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise _refuse("destination_conflict", object_id=object_id)
    hasher, total = hashlib.sha256(), 0
    with open(path, "rb") as reader:
        while chunk := reader.read(min(_CHUNK, size + 1 - total)):
            hasher.update(chunk)
            total += len(chunk)
            if total > size:
                raise _refuse("object_changed", object_id=object_id)
    if total != size or hasher.hexdigest() != digest:
        raise _refuse("object_changed", object_id=object_id)
    return True


def _open_source(path, object_id: str):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise _refuse("object_missing", object_id=object_id) from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise _refuse("object_changed", object_id=object_id)
    return os.fdopen(fd, "rb")


def _copy(source, destination, digest: str, size: int, object_id: str) -> None:
    ensure_directory(destination.parent)
    part = destination.with_name(f"{destination.name}.{os.getpid()}.{time.monotonic_ns()}.part")
    try:
        hasher, total = hashlib.sha256(), 0
        with _open_source(source, object_id) as reader:
            out = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(out, "wb") as writer:
                while chunk := reader.read(min(_CHUNK, size + 1 - total)):
                    hasher.update(chunk)
                    total += len(chunk)
                    if total > size:
                        raise _refuse("object_changed", object_id=object_id)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        if total != size or hasher.hexdigest() != digest:
            raise _refuse("object_changed", object_id=object_id)
        try:
            os.link(part, destination)
        except FileExistsError:
            raise _refuse("destination_conflict", object_id=object_id) from None
        fsync_directory(destination.parent)
    finally:
        try:
            os.unlink(part)
        except FileNotFoundError:
            pass


def admit_closure_objects(workspace) -> dict:
    """Copy exactly the pinned snapshot's fragment objects from the read-only source store."""
    try:
        return _admit(workspace)
    except (OpsError, DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc


def _digest(object_id, content_hash, size, storage_key) -> str:
    """The sha256 hex of a well-formed object row; anything else is refused."""
    match = _HASH.match(content_hash) if isinstance(content_hash, str) else None
    if (len(object_id) > 200 or len(storage_key) > 200 or match is None
            or type(size) is not int or size < 0
            or storage_key != f"objects/{match.group(1)[:2]}/{match.group(1)}"):
        raise _refuse("object_ref_invalid", object_id=object_id[:80])
    return match.group(1)


def _admit(workspace) -> dict:
    try:
        snapshot_id, conn = workspace.snapshot_id, workspace.conn
        source_root, target_root = workspace._source_store.root, workspace.store.root
    except AttributeError:
        raise _refuse("workspace_invalid") from None
    if not isinstance(snapshot_id, str) or conn.execute(
            "SELECT 1 FROM data_snapshot_tables WHERE snapshot_id = ? LIMIT 1",
            (snapshot_id,)).fetchone() is None:
        raise _refuse("workspace_invalid")
    if conn.execute("SELECT 1 FROM data_raw_receipts LIMIT 1").fetchone() is not None:
        raise _refuse("receipts_present")
    workspace.source_snapshot()
    started = last = time.monotonic()
    _log("object admission start")
    copied = present = copied_bytes = total_bytes = index = 0
    for index, (object_id, content_hash, size, storage_key) in enumerate(
            conn.execute(_NAMED, (snapshot_id,)), 1):
        if index > _MAX_OBJECTS:
            raise _refuse("objects_over_cap")
        digest = _digest(object_id, content_hash, size, storage_key)
        total_bytes += size
        if total_bytes > _MAX_BYTES:
            raise _refuse("bytes_over_cap", object_id=object_id)
        destination = target_root / storage_key
        if _present(destination, size, digest, object_id):
            present += 1
        else:
            _copy(source_root / storage_key, destination, digest, size, object_id)
            copied += 1
            copied_bytes += size
        if time.monotonic() - last >= _PROGRESS_SECONDS:
            last = time.monotonic()
            _log(f"{index} objects, {copied} copied, {last - started:.0f}s elapsed")
    workspace.source_snapshot()
    _log(f"object admission end copied={copied} present={present} bytes={copied_bytes} "
         f"{time.monotonic() - started:.0f}s elapsed")
    return {"snapshot_id": snapshot_id, "objects": index, "copied": copied,
            "present": present, "bytes": copied_bytes}
