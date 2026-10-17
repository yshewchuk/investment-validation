"""Content-addressed object layout, durable writes and the bounded regular-file read for a staged release."""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from engine.v2.foundation import ensure_directory, fsync_directory
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state

DEPLOYMENT_DIR = "deployment"
OBJECTS_DIR = "objects"


class ReleaseLayoutError(ValueError):
    """The staged release root is malformed."""


class ObjectVerificationError(ReleaseLayoutError):
    """Existing or published object bytes failed bounded verification."""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def object_relpath(member_hash: str) -> str:
    """The deployment store's own content-addressed object path."""
    return f"{OBJECTS_DIR}/" + member_hash.removeprefix("sha256:")


def deployment_root(release_root: Path) -> Path:
    return Path(release_root) / DEPLOYMENT_DIR


def write_object(release_root: Path, payload: bytes) -> tuple[str, str]:
    """Durably create or verify exact bytes, never replace an existing object."""
    digest = sha256_bytes(payload)
    rel = object_relpath(digest)
    dest = deployment_root(release_root) / rel
    try:
        ensure_directory(dest.parent)
        if not dest.is_file():
            scratch = deployment_root(release_root) / "tmp"
            ensure_directory(scratch)
            with tempfile.TemporaryDirectory(prefix="phase5-object-", dir=scratch) as directory:
                tmp = Path(directory) / "payload"
                with tmp.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(tmp, dest)
                except FileExistsError:
                    if not dest.is_file():
                        raise
        try:
            actual = read_bounded_state(dest, len(payload), no_follow=True)
        except (BoundedStateReadError, OSError) as exc:
            raise ObjectVerificationError("phase5 object bytes could not be verified") from exc
        if actual != payload:
            raise ObjectVerificationError("phase5 object bytes differ from payload")
        fsync_directory(dest.parent)
        fsync_directory(dest.parent.parent)
    except OSError as exc:
        raise ReleaseLayoutError("phase5 object could not be published") from exc
    return digest, rel


def read_regular_file(path: Path, limit: int) -> bytes | None:
    """Bytes of a regular file opened no-follow: None when absent; ReleaseLayoutError when not a
    regular file, when longer than ``limit``, or when ``limit`` is not a non-negative integer; a
    symlink raises OSError (callers wrap it); a path containing a NUL byte raises ValueError from
    os.open (callers wrap it)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ReleaseLayoutError("read limit must be a non-negative integer")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ReleaseLayoutError("phase5 catalog is not a bounded regular file")
        handle = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ReleaseLayoutError("phase5 catalog is not a bounded regular file")
    return data


__all__ = [
    "DEPLOYMENT_DIR", "OBJECTS_DIR", "ObjectVerificationError", "ReleaseLayoutError",
    "deployment_root", "object_relpath", "read_regular_file", "sha256_bytes",
    "write_object",
]
