"""Durable nightly session/generation identity and compare-and-swap state.

Slice 1 of the nightly entry-point design (#564); contract in
``engine/v2/ops/ARCHITECTURE.md`` ("Nightly session identity"). A leaf: it is not
yet called by ``nightly_trigger`` and runs no step. The old per-date
``<as_of>.json`` receipts live at another path and are never read here.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.v2.ops.errors import fail

__all__ = ["Generation", "SessionIdentity", "SessionState", "atomic_write", "compare_and_swap",
           "ensure_session", "load_session", "mark_started", "request_rerun",
           "session_path", "session_transition_lock"]

SCHEMA = "nightly_session.v1"
KEY_VERSION = "nightly_session_key.v1"
STATE_DIR = ("reports", "phase6", "nightly_trigger", "sessions")
MAX_SWAP_ATTEMPTS = 3
MAX_SESSION_BYTES = 1024 * 1024
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _require_regular_fd(fd: int, description: str) -> os.stat_result:
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", f"{description} is unreadable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise fail("INTEGRITY_FAILED", f"{description} is not a regular file")
    return info


def _open_regular(path: Path, flags: int, description: str,
                  *, missing_ok: bool = False, mode: int = 0o666) -> int | None:
    try:
        fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise fail("INTEGRITY_FAILED", f"{description} is missing") from exc
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", f"{description} is unsafe or unreadable") from exc
    try:
        _require_regular_fd(fd, description)
    except Exception:
        _close_publication_fd(
            fd, f"{description} close failed", preserve_error=True)
        raise
    return fd


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SessionIdentity:
    as_of: str
    scope: str
    selection_identity: str
    catalog_identity: str

    def __post_init__(self) -> None:
        fields = (self.as_of, self.scope, self.selection_identity, self.catalog_identity)
        if not all(isinstance(f, str) and f.strip() for f in fields) or not _DATE.fullmatch(
                self.as_of):
            raise fail("INVALID_REQUEST", "session identity fields must be non-empty")
        try:
            date.fromisoformat(self.as_of)
        except ValueError as exc:
            raise fail("INVALID_REQUEST", "as_of is not a calendar date") from exc

    @property
    def session_key(self) -> str:
        return _digest(KEY_VERSION, self.as_of, self.scope, self.selection_identity,
                       self.catalog_identity)


@dataclass(frozen=True)
class Generation:
    generation: int
    run_id: str
    reason: str  # "initial" | "rerun"
    status: str  # "allocated" | "started"
    invalidation: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionState:
    identity: SessionIdentity
    revision: int
    generations: tuple[Generation, ...]

    @property
    def active(self) -> Generation:
        return self.generations[-1]


def session_path(root: Path, identity: SessionIdentity) -> Path:
    return Path(root).joinpath(*STATE_DIR, f"{identity.session_key}.json")


def _encode(state: SessionState) -> str:
    identity = state.identity
    return json.dumps({
        "schema": SCHEMA, "session_key": identity.session_key, "revision": state.revision,
        "identity": {"as_of": identity.as_of, "scope": identity.scope,
                     "selection_identity": identity.selection_identity,
                     "catalog_identity": identity.catalog_identity},
        "generations": [{"generation": g.generation, "run_id": g.run_id,
                         "reason": g.reason, "status": g.status,
                         "invalidation": list(g.invalidation)} for g in state.generations],
    }, sort_keys=True)


def _generations_valid(gens: tuple[Generation, ...], key: str) -> bool:
    return bool(gens) and all(
        type(g.generation) is int and g.generation == i and g.run_id == _digest(key, i)
        and g.status in ("allocated", "started") and g.reason == ("initial" if i == 1 else "rerun")
        and all(isinstance(s, str) for s in g.invalidation)
        and (g.status == "started" or i == len(gens))
        for i, g in enumerate(gens, 1))


def _parse_generation(g: dict) -> Generation:
    raw = g["invalidation"]
    return Generation(g["generation"], g["run_id"], g["reason"], g["status"],
                      tuple(raw) if isinstance(raw, list) else (None,))


def _decode(text: str, identity: SessionIdentity) -> SessionState:
    try:
        doc = json.loads(text)
        if not isinstance(doc, dict):
            raise ValueError("not an object")
        if doc.get("schema") != SCHEMA:
            raise fail("CHECKPOINT_INCOMPATIBLE",
                       "session state has another schema; move the file aside")
        key = identity.session_key
        gens = tuple(_parse_generation(g) for g in doc["generations"])
        ok = (doc["session_key"] == key and type(doc["revision"]) is int
              and doc["identity"] == {
                  "as_of": identity.as_of, "scope": identity.scope,
                  "selection_identity": identity.selection_identity,
                  "catalog_identity": identity.catalog_identity}
              and doc["revision"] >= 1 and _generations_valid(gens, key))
    except (ValueError, KeyError, TypeError) as exc:
        raise fail("INTEGRITY_FAILED", "session state is unreadable") from exc
    if not ok:
        raise fail("INTEGRITY_FAILED", "session state failed identity checks")
    return SessionState(identity, doc["revision"], gens)


def load_session(root: Path, identity: SessionIdentity) -> SessionState | None:
    """The recorded state, ``None`` only when no file exists; never ``None`` for bad bytes."""
    path = session_path(root, identity)
    fd = _open_regular(path, os.O_RDONLY, "session state", missing_ok=True)
    if fd is None:
        return None
    try:
        info = _require_regular_fd(fd, "session state")
        if info.st_size > MAX_SESSION_BYTES:
            raise fail("INTEGRITY_FAILED", "session state is too large")
        chunks: list[bytes] = []
        remaining = MAX_SESSION_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_SESSION_BYTES:
            raise fail("INTEGRITY_FAILED", "session state is too large")
        text = data.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "session state is unreadable") from exc
    finally:
        active_error = sys.exc_info()[0] is not None
        _close_publication_fd(
            fd, "session state close failed", preserve_error=active_error)
    return _decode(text, identity)


def _create_atomic_temp(path: Path, dir_fd: int) -> tuple[int, str]:
    temporary_name = path.name + ".tmp"
    try:
        info = os.stat(temporary_name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "temporary session file is unreadable") from exc
    else:
        if not stat.S_ISREG(info.st_mode):
            raise fail("INTEGRITY_FAILED", "temporary session file is not regular")
        try:
            os.unlink(temporary_name, dir_fd=dir_fd)
        except OSError as exc:
            raise fail("INTEGRITY_FAILED", "stale temporary session file cannot be removed") from exc
    try:
        fd = os.open(temporary_name,
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o666, dir_fd=dir_fd)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "temporary session file is unsafe or already exists") from exc
    try:
        info = os.fstat(fd)
    except OSError as exc:
        typed = fail("INTEGRITY_FAILED", "temporary session file is unreadable")
        typed.__cause__ = exc
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name, dir_fd=dir_fd)
        except OSError:
            pass
        raise typed
    if not stat.S_ISREG(info.st_mode):
        typed = fail("INTEGRITY_FAILED", "temporary session file is not regular")
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_name, dir_fd=dir_fd)
        except OSError:
            pass
        raise typed
    return fd, temporary_name


def _open_publication_directory(parent: Path) -> int:
    try:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session publication directory is unsafe") from exc
    try:
        info = os.fstat(fd)
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise fail("INTEGRITY_FAILED", "session publication directory is unreadable") from exc
    if not stat.S_ISDIR(info.st_mode):
        try:
            os.close(fd)
        except OSError:
            pass
        raise fail("INTEGRITY_FAILED", "session publication path is not a directory")
    return fd


def _check_publication_destination(dir_fd: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session publication destination is unreadable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise fail("INTEGRITY_FAILED", "session publication destination is not regular")


def _write_and_sync_temp(fd: int, data: bytes) -> None:
    try:
        handle = os.fdopen(fd, "wb")
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        raise fail("INTEGRITY_FAILED", "session state write failed") from exc
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session state write failed") from exc


def _replace_publication_temp(dir_fd: int, temporary_name: str, destination_name: str) -> None:
    try:
        os.replace(temporary_name, destination_name,
                   src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session publication destination is unsafe") from exc


def _sync_publication_directory(dir_fd: int) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session publication directory fsync failed") from exc


def _close_publication_fd(fd: int, message: str, preserve_error: bool = False) -> None:
    try:
        os.close(fd)
    except OSError as exc:
        if not preserve_error:
            raise fail("INTEGRITY_FAILED", message) from exc


def _cleanup_publication_temp(dir_fd: int, fd: int, temporary_name: str,
                              temp_created: bool, published: bool,
                              preserve_error: bool = False) -> None:
    if fd != -1:
        _close_publication_fd(fd, "session publication file close failed", preserve_error)
    if temp_created and not published:
        try:
            os.unlink(temporary_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if not preserve_error:
                raise fail("INTEGRITY_FAILED", "session publication cleanup failed") from exc


def atomic_write(path: Path, text: str, *, max_bytes: int | None = None,
                 size_error: str = "publication data is too large") -> None:
    """Durably replace ``path``, optionally enforcing its owner's size limit."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise fail("INVALID_REQUEST", "publication byte limit must be a nonnegative integer")
    data = text.encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        raise fail("INTEGRITY_FAILED", size_error)
    dir_fd = _open_publication_directory(path.parent)
    published = False
    temp_created = False
    tmp_fd = -1
    tmp_name = path.name + ".tmp"
    try:
        _check_publication_destination(dir_fd, path.name)
        tmp_fd, tmp_name = _create_atomic_temp(path, dir_fd)
        temp_created = True
        write_fd = tmp_fd
        tmp_fd = -1
        _write_and_sync_temp(write_fd, data)
        _replace_publication_temp(dir_fd, tmp_name, path.name)
        published = True
        _sync_publication_directory(dir_fd)
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            _cleanup_publication_temp(
                dir_fd, tmp_fd, tmp_name, temp_created, published, active_error)
        finally:
            _close_publication_fd(
                dir_fd, "session publication directory close failed",
                active_error or sys.exc_info()[0] is not None)


@contextmanager
def session_transition_lock(root: Path, identity: SessionIdentity):
    """Hold the per-session lock used for transitions and run-scoped publications."""
    if not isinstance(identity, SessionIdentity):
        raise fail("INVALID_REQUEST", "session transition lock needs a session identity")
    path = session_path(root, identity)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED",
                   "session state directory is unsafe or unavailable") from exc
    try:
        lock_fd = _open_regular(path.with_name(path.name + ".lock"),
                                os.O_RDWR | os.O_APPEND | os.O_CREAT, "session lock")
        assert lock_fd is not None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                os.close(lock_fd)
            except OSError as exc:
                if not active_error:
                    raise fail("INTEGRITY_FAILED",
                               "session transition lock close failed") from exc
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "session transition lock is unavailable") from exc


def compare_and_swap(root: Path, identity: SessionIdentity, expected_revision: int,
                     generations: tuple[Generation, ...]) -> SessionState | None:
    """Write ``generations`` at ``expected_revision + 1`` iff the stored revision is
    ``expected_revision`` (0 = no file). Returns ``None`` when the swap was lost."""
    path = session_path(root, identity)
    with session_transition_lock(root, identity):
        current = load_session(root, identity)
        if (current.revision if current else 0) != expected_revision:
            return None
        state = SessionState(identity, expected_revision + 1, generations)
        _decode(_encode(state), identity)  # the loader's own rules; raises before any write
        atomic_write(path, _encode(state), max_bytes=MAX_SESSION_BYTES,
                     size_error="session state is too large")
        return state


def _generation(identity: SessionIdentity, number: int, reason: str,
                invalidation: tuple[str, ...] = ()) -> Generation:
    return Generation(number, _digest(identity.session_key, number), reason, "allocated",
                      tuple(sorted(set(invalidation))))


def _swap_loop(root: Path, identity: SessionIdentity, step):
    """``step(state | None) -> (result, new generations | None)``; a ``None`` new value
    means the loaded state already answers the request."""
    for _ in range(MAX_SWAP_ATTEMPTS):
        state = load_session(root, identity)
        result, generations = step(state)
        if generations is None:
            return result
        won = compare_and_swap(root, identity, state.revision if state else 0, generations)
        if won is not None:
            return won
    raise fail("LEASE_LOST", "session state changed under concurrent callers")


def ensure_session(root: Path, identity: SessionIdentity) -> SessionState:
    """Ordinary invocation: the one derived session, created at generation 1 on first use."""
    return _swap_loop(root, identity, lambda s: (s, None) if s else (
        None, (_generation(identity, 1, "initial"),)))


def request_rerun(root: Path, identity: SessionIdentity,
                  invalidation: tuple[str, ...] = ()) -> SessionState:
    """Explicit rerun: allocate the next generation, or reattach to one still ``allocated``."""
    if not isinstance(invalidation, (tuple, list)) or not all(
            isinstance(s, str) for s in invalidation):
        raise fail("INVALID_REQUEST", "invalidation must be a sequence of step-name strings")
    def step(state):
        if state is None:
            raise fail("INVALID_REQUEST", "no session exists to rerun")
        if state.active.status == "allocated":
            return state, None
        return None, state.generations + (
            _generation(identity, state.active.generation + 1, "rerun", invalidation),)
    return _swap_loop(root, identity, step)


def mark_started(root: Path, identity: SessionIdentity, generation: int) -> SessionState:
    """Move the active generation from ``allocated`` to ``started`` (idempotent)."""
    def step(state):
        if type(generation) is not int:
            raise fail("INVALID_REQUEST", "generation must be an integer")
        if state is None or state.active.generation != generation:
            raise fail("CHECKPOINT_INCOMPATIBLE", "generation is not the active one")
        if state.active.status == "started":
            return state, None
        started = Generation(state.active.generation, state.active.run_id, state.active.reason,
                             "started", state.active.invalidation)
        return None, state.generations[:-1] + (started,)
    return _swap_loop(root, identity, step)