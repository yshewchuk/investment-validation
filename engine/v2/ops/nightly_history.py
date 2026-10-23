"""Pinned D-1 history import for the nightly snapshot chain (slice 5 of #564)."""
from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from engine.v2.ops import ledger_history_import
from engine.v2.ops.errors import fail
from engine.v2.ops.ledger_history_import import import_history
from engine.v2.ops.nightly_chain import ChainResult, ChainStep, run_snapshot_chain
from engine.v2.ops.nightly_receipts import Effect
from engine.v2.ops.nightly_session import (
    STATE_DIR,
    SessionIdentity,
    atomic_write,
    load_session,
)

__all__ = ["run_pinned_history_import"]

INPUTS_SCHEMA = "nightly_history_inputs.v1"
RECEIPT_SCHEMA = "nightly_history_receipt.v1"


@contextmanager
def _history_lock(root: Path, identity: SessionIdentity):
    path = Path(root).joinpath(*STATE_DIR, f"{identity.session_key}.history.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _pins(source_root: Path) -> tuple[tuple[str, str], ...]:
    if not (source_root / "ledger").is_dir():
        raise fail("SOURCE_NOT_FOUND", "source root has no ledger directory")
    discovered = ledger_history_import.discover_history_files(source_root)
    return tuple(sorted(
        (relative, hashlib.sha256(
            ledger_history_import._read_pinned_bytes(source_root, path)).hexdigest())
        for relative, path in discovered.items()))


def _valid_pin_entries(files):
    return all(
        isinstance(pair, list) and len(pair) == 2
        and isinstance(pair[0], str) and isinstance(pair[1], str)
        and len(pair[1]) == 64
        and all(char in "0123456789abcdef" for char in pair[1])
        for pair in files)


def _read_pins(path, run_id, through):
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable") from exc
    if not isinstance(document, dict):
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable")
    if (document.get("schema") != INPUTS_SCHEMA
            or document.get("run_id") != run_id
            or document.get("through") != through
            or not isinstance(document.get("files"), list)):
        raise fail("INTEGRITY_FAILED", "history file pins failed identity checks")
    if not _valid_pin_entries(document["files"]):
        raise fail("INTEGRITY_FAILED", "history file pins are malformed")
    return tuple(tuple(pair) for pair in document["files"])


def _write_inputs(path, run_id, through, pins):
    document = {"schema": INPUTS_SCHEMA, "run_id": run_id, "through": through,
                "files": [list(pair) for pair in pins]}
    atomic_write(path, json.dumps(document, sort_keys=True))


def _load_or_dry_run_pins(root, identity, generation, source_root, conn, clock):
    state = load_session(root, identity)
    if (state is None or type(generation) is not int or state.active.generation != generation
            or state.active.status != "started"):
        raise fail("CHECKPOINT_INCOMPATIBLE", "generation is not the active started one")
    through = (date.fromisoformat(identity.as_of) - timedelta(days=1)).isoformat()
    path = Path(root).joinpath(*STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    pins = _read_pins(path, state.active.run_id, through)
    if pins is None:
        pins = _pins(Path(source_root))
        import_history(conn, root, source_root, through=date.fromisoformat(through),
                       dry_run=True, clock=clock, pinned_files=pins)
        _write_inputs(path, state.active.run_id, through, pins)
    return state.active.run_id, through, pins


def _artifact_successor(path: Path, digest: str, predecessor: str) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", "nightly artifact cannot be read") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise fail("INTEGRITY_FAILED", "nightly artifact differs from its pinned receipt")
    return predecessor


def run_pinned_history_import(root: Path, identity: SessionIdentity, generation: int,
                              start_snapshot: str, *, conn, source_root, clock) -> ChainResult:
    """Import pinned D-1 ledger history from the preceding price-history snapshot."""
    source_root = Path(source_root)
    with _history_lock(root, identity):
        run_id, through, pins = _load_or_dry_run_pins(
            root, identity, generation, source_root, conn, clock)
        history_path = Path(root).joinpath(*STATE_DIR, f"{run_id}.history-complete.json")
        history_text = json.dumps({"schema": RECEIPT_SCHEMA, "run_id": run_id,
                                   "through": through,
                                   "files": [list(pair) for pair in pins]}, sort_keys=True)
        history_hash = hashlib.sha256(history_text.encode()).hexdigest()

        def history_effect(_predecessor):
            return Effect("artifact", str(history_path), history_hash)

        def import_pinned_history(_predecessor):
            import_history(conn, root, source_root, through=date.fromisoformat(through),
                           dry_run=False, clock=clock, pinned_files=pins)
            atomic_write(history_path, history_text)
        step = ChainStep(
            "history_import", history_effect, import_pinned_history,
            lambda predecessor: _artifact_successor(history_path, history_hash, predecessor))
        return run_snapshot_chain(root, identity, generation, start_snapshot, [step], conn=conn)
