"""Durable, content-addressed artifact publication — phase-1 guide §7.1.

A worker writes candidates under ``attempts/<attempt_id>/staging/``. Nothing it
writes is trusted. The coordinator opens each candidate **without following a
symlink at any path component**, refuses anything but a regular file with a
single link (a hard link back into a production directory is not a private
copy), and **copies** the bytes into its own temporary file while hashing them.

Copying rather than renaming the worker's file is deliberate. A stale worker
that still holds the file open can keep writing after a rename, and the
published bytes would then differ from the bytes that were hashed. The copy is
what the hash describes.

The copy is fsynced, made read-only, and hard-linked into its content-addressed
name. ``link`` fails if the name exists, so two publishers of the same bytes
cannot race each other into a torn file; an existing object is re-verified
instead of trusted. The parent directory is fsynced before the reference is
returned, so by the time a catalog row can point at an artifact, the artifact
is durable. A crash can leave an unreferenced object or temporary file behind,
which is safe; it can never leave a reference to partial bytes.

Orphans are not deleted here. Garbage collection needs catalog reachability,
leases and retention rules (§7.1), and belongs to whoever owns those.

``fault`` is a named-crash-point hook for the O11 fault-injection tests; it is
``None`` in production and costs one call per publication.
"""
from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation.canonical import CONTENT_HASH_PREFIX, content_hash

__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "ensure_directory",
    "fsync_directory",
    "safe_relative_path",
]

_CHUNK = 1 << 20
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
#: O_NONBLOCK so a FIFO planted as a candidate is refused, not blocked on.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


class ArtifactError(Exception):
    """A refused or failed artifact operation, with a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def safe_relative_path(rel: str) -> tuple[str, ...]:
    """Split a worker-supplied relative path, refusing every escape."""
    if not isinstance(rel, str) or not rel or "\x00" in rel or "\\" in rel:
        raise ArtifactError("UNSAFE_PATH", "not a plain relative path")
    parts = tuple(rel.split("/"))
    if rel.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ArtifactError("UNSAFE_PATH", "absolute, empty-segment or traversing path")
    return parts


def fsync_directory(path: Path) -> None:
    """Make a directory's entries durable; a renamed or linked file is not durable until this."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: Path) -> None:
    """Create ``path`` and missing parents, syncing the parent of each new entry."""
    missing: list[Path] = []
    probe = Path(path)
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        fsync_directory(directory.parent)


def _open_beneath(base: Path, parts: tuple[str, ...]) -> int:
    """Open ``base/parts...`` read-only, refusing a symlink at every component."""
    try:
        fd = os.open(base, _DIR_FLAGS)
    except OSError as exc:
        raise ArtifactError("UNSAFE_PATH", f"{base} is not a real directory") from exc
    try:
        for part in parts[:-1]:
            nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], _FILE_FLAGS, dir_fd=fd)
    except FileNotFoundError as exc:
        raise ArtifactError("MISSING", "/".join(parts)) from exc
    except OSError as exc:
        raise ArtifactError("UNSAFE_PATH", f"{'/'.join(parts)}: {exc.strerror}") from exc
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _hash_fd(fd: int, sink: int | None = None,
             max_bytes: int | None = None) -> tuple[str, int]:
    """Stream ``fd``, optionally copying into ``sink``. Returns ``(hexdigest, size)``."""
    digest, size = hashlib.sha256(), 0
    while chunk := os.read(fd, _CHUNK):
        size += len(chunk)
        if max_bytes is not None and size > max_bytes:
            raise ArtifactError("SIZE_LIMIT", f"candidate exceeds {max_bytes} bytes")
        digest.update(chunk)
        if sink is not None:
            _write_all(sink, chunk)
    return digest.hexdigest(), size


class ArtifactStore:
    """Immutable objects under ``<root>/objects/``, staged work under ``<root>/attempts/``."""

    def __init__(self, root: Path | str, *, fault: Callable[[str], None] | None = None) -> None:
        self.root = Path(root).resolve()
        self._fault = fault or (lambda point: None)

    # -- staging ------------------------------------------------------------

    def staging_dir(self, attempt_id: str) -> Path:
        """The one directory an attempt may write candidates into."""
        _one_segment(attempt_id)
        path = self.root / "attempts" / attempt_id / "staging"
        ensure_directory(path)
        return path

    def publish_candidate(self, attempt_id: str, rel: str, *, schema_ref: str,
                          max_bytes: int | None = None) -> ArtifactRef:
        """Copy one staged candidate into the immutable store."""
        _one_segment(attempt_id)
        parts = ("attempts", attempt_id, "staging", *safe_relative_path(rel))
        src = _open_beneath(self.root, parts)
        try:
            info = os.fstat(src)
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactError("NOT_A_REGULAR_FILE", rel)
            if info.st_nlink != 1:
                raise ArtifactError("UNSAFE_PATH", f"{rel} has {info.st_nlink} links; "
                                    "a candidate must be a private copy")
            os.set_blocking(src, True)
            return self._commit(lambda out: _hash_fd(src, out, max_bytes), schema_ref)
        finally:
            os.close(src)

    def publish_bytes(self, data: bytes, *, schema_ref: str) -> ArtifactRef:
        """Publish coordinator-produced bytes (a manifest, a receipt)."""
        def write(out: int) -> tuple[str, int]:
            _write_all(out, data)
            return hashlib.sha256(data).hexdigest(), len(data)
        return self._commit(write, schema_ref)

    # -- reading ------------------------------------------------------------

    def verify(self, ref: ArtifactRef) -> Path:
        """Re-hash an object before it is trusted; return its path if it matches."""
        parts = self._object_parts(ref)
        fd = _open_beneath(self.root, parts)
        try:
            self._check_fd(fd, ref)
        finally:
            os.close(fd)
        return self.root.joinpath(*parts)

    def read_verified(self, ref: ArtifactRef) -> bytes:
        """Read an object's bytes, hashing exactly the bytes returned."""
        fd = _open_beneath(self.root, self._object_parts(ref))
        try:
            os.set_blocking(fd, True)
            data = b"".join(iter(lambda: os.read(fd, _CHUNK), b""))
        finally:
            os.close(fd)
        if len(data) != ref.byte_size or \
                CONTENT_HASH_PREFIX + hashlib.sha256(data).hexdigest() != ref.content_hash:
            raise ArtifactError("INTEGRITY_FAILED", f"{ref.storage_key} does not match its reference")
        return data

    # -- internals ----------------------------------------------------------

    def _commit(self, write: Callable[[int], tuple[str, int]], schema_ref: str) -> ArtifactRef:
        tmp_dir = self.root / "tmp"
        ensure_directory(tmp_dir)
        tmp = tmp_dir / f"{uuid.uuid4().hex}.part"
        out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            hexdigest, size = write(out)
            os.fsync(out)
        except BaseException:
            os.close(out)
            tmp.unlink(missing_ok=True)
            raise
        os.close(out)
        os.chmod(tmp, 0o444)
        self._fault("copied")
        ref = _reference(hexdigest, size, schema_ref)
        dest_dir = self.root / "objects" / hexdigest[:2]
        ensure_directory(dest_dir)
        try:
            os.link(tmp, dest_dir / hexdigest)
        except FileExistsError:
            self.verify(ref)
        fsync_directory(dest_dir)
        self._fault("linked")
        tmp.unlink()
        return ref

    @staticmethod
    def _object_parts(ref: ArtifactRef) -> tuple[str, ...]:
        parts = safe_relative_path(ref.storage_key)
        digest = ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
        if (len(parts) != 3 or parts[0] != "objects" or parts[2] != digest
                or parts[1] != digest[:2]):
            raise ArtifactError("INTEGRITY_FAILED", "storage key does not match content hash")
        return parts

    @staticmethod
    def _check_fd(fd: int, ref: ArtifactRef) -> None:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError("INTEGRITY_FAILED", f"{ref.storage_key} is not a regular file")
        os.set_blocking(fd, True)
        hexdigest, size = _hash_fd(fd)
        if size != ref.byte_size or CONTENT_HASH_PREFIX + hexdigest != ref.content_hash:
            raise ArtifactError("INTEGRITY_FAILED", f"{ref.storage_key} does not match its reference")


def _one_segment(attempt_id: str) -> None:
    if len(safe_relative_path(attempt_id)) != 1:
        raise ArtifactError("UNSAFE_PATH", "an attempt id is exactly one path segment")


def _reference(hexdigest: str, size: int, schema_ref: str) -> ArtifactRef:
    """Deterministic identity: the same bytes under the same schema are one artifact."""
    chash = CONTENT_HASH_PREFIX + hexdigest
    ident = content_hash({"content_hash": chash, "schema_ref": schema_ref})
    return ArtifactRef(
        artifact_id="art_" + ident.removeprefix(CONTENT_HASH_PREFIX)[:32],
        content_hash=chash,
        schema_ref=schema_ref,
        byte_size=size,
        storage_key=f"objects/{hexdigest[:2]}/{hexdigest}",
    )
