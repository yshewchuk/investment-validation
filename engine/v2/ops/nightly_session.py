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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.v2.ops.errors import fail

__all__ = ["Generation", "SessionIdentity", "SessionState", "atomic_write", "compare_and_swap",
           "ensure_session", "load_session", "mark_started", "request_rerun",
           "session_path"]

SCHEMA = "nightly_session.v1"
KEY_VERSION = "nightly_session_key.v1"
STATE_DIR = ("reports", "phase6", "nightly_trigger", "sessions")
MAX_SWAP_ATTEMPTS = 3
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


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
    try:
        text = session_path(root, identity).read_text()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "session state is unreadable") from exc
    return _decode(text, identity)


def atomic_write(path: Path, text: str) -> None:
    """Durably replace ``path`` with ``text``: temp file, fsync, rename, directory fsync."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def compare_and_swap(root: Path, identity: SessionIdentity, expected_revision: int,
                     generations: tuple[Generation, ...]) -> SessionState | None:
    """Write ``generations`` at ``expected_revision + 1`` iff the stored revision is
    ``expected_revision`` (0 = no file). Returns ``None`` when the swap was lost."""
    path = session_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = load_session(root, identity)
        if (current.revision if current else 0) != expected_revision:
            return None
        state = SessionState(identity, expected_revision + 1, generations)
        _decode(_encode(state), identity)  # the loader's own rules; raises before any write
        atomic_write(path, _encode(state))
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