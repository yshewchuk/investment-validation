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

INPUTS_SCHEMA = "nightly_history_inputs.v1.1"
LEGACY_INPUTS_SCHEMA = "nightly_history_inputs.v1"
RECEIPT_SCHEMA = "nightly_history_receipt.v1.1"
LEGACY_RECEIPT_SCHEMA = "nightly_history_receipt.v1"


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


def _valid_coverage(coverage, through):
    if not isinstance(coverage, dict) or coverage.get("through") != through:
        return False
    families = coverage.get("families")
    if not isinstance(families, dict):
        return False
    for entry in families.values():
        if not isinstance(entry, dict):
            return False
        rows, status = entry.get("rows"), entry.get("status")
        if type(rows) is not int or rows < 0:
            return False
        if status not in ("present", "no_rows") or (status == "present") != (rows > 0):
            return False
    return True


def _read_pins(path, run_id, through):
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable") from exc
    if not isinstance(document, dict):
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable")
    if (document.get("schema") not in (LEGACY_INPUTS_SCHEMA, INPUTS_SCHEMA)
            or document.get("run_id") != run_id
            or document.get("through") != through
            or not isinstance(document.get("files"), list)):
        raise fail("INTEGRITY_FAILED", "history file pins failed identity checks")
    if not _valid_pin_entries(document["files"]):
        raise fail("INTEGRITY_FAILED", "history file pins are malformed")
    coverage = document.get("coverage")
    if coverage is not None and not _valid_coverage(coverage, through):
        raise fail("INTEGRITY_FAILED", "history coverage failed identity checks")
    return tuple(tuple(pair) for pair in document["files"]), coverage


def _write_inputs(path, run_id, through, pins, coverage):
    document = {"schema": INPUTS_SCHEMA, "run_id": run_id, "through": through,
                "files": [list(pair) for pair in pins], "coverage": coverage}
    atomic_write(path, json.dumps(document, sort_keys=True))


def _legacy_receipt_text(path, run_id, through, pins):
    """An already-completed v1 receipt for exactly this run and pin set.

    v1 predates ``coverage``.  Reusing its bytes (rather than minting a v1.1
    effect) keeps the recorded step receipt, artifact hash and this run's
    effect in agreement, so resume skips a completed import instead of
    failing ``IDEMPOTENCY_CONFLICT`` or silently trusting another effect.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "nightly history receipt is unreadable") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise fail("INTEGRITY_FAILED", "nightly history receipt is unreadable") from exc
    if not isinstance(document, dict) or document.get("schema") != LEGACY_RECEIPT_SCHEMA:
        return None
    files = document.get("files")
    if (document.get("run_id") != run_id or document.get("through") != through
            or not isinstance(files, list) or not _valid_pin_entries(files)
            or tuple(tuple(pair) for pair in files) != pins):
        raise fail("INTEGRITY_FAILED", "legacy history receipt failed identity checks")
    return text


def _load_or_dry_run_pins(root, identity, generation, source_root, conn, clock):
    state = load_session(root, identity)
    if (state is None or type(generation) is not int or state.active.generation != generation
            or state.active.status != "started"):
        raise fail("CHECKPOINT_INCOMPATIBLE", "generation is not the active started one")
    through = (date.fromisoformat(identity.as_of) - timedelta(days=1)).isoformat()
    path = Path(root).joinpath(*STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    loaded = _read_pins(path, state.active.run_id, through)
    if loaded is None:
        pins, coverage = _pins(Path(source_root)), None
    else:
        pins, coverage = loaded
    if coverage is None:
        summary = import_history(conn, root, source_root, through=date.fromisoformat(through),
                                 dry_run=True, clock=clock, pinned_files=pins)
        coverage = summary["coverage"]
        _write_inputs(path, state.active.run_id, through, pins, coverage)
    return state.active.run_id, through, pins, coverage


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
        run_id, through, pins, coverage = _load_or_dry_run_pins(
            root, identity, generation, source_root, conn, clock)
        history_path = Path(root).joinpath(*STATE_DIR, f"{run_id}.history-complete.json")
        history_text = _legacy_receipt_text(history_path, run_id, through, pins)
        if history_text is None:
            history_text = json.dumps({"schema": RECEIPT_SCHEMA, "run_id": run_id,
                                       "through": through,
                                       "files": [list(pair) for pair in pins],
                                       "coverage": coverage}, sort_keys=True)
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
