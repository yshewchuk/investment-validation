"""Pinned history bootstrap and final input capture for the nightly chain (slice 5 of #564)."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from engine.v2.foundation import to_document
from engine.v2.ops import capture_inputs
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.ledger_history_import import import_history
from engine.v2.ops.nightly_chain import ChainResult, ChainStep, run_snapshot_chain
from engine.v2.ops.nightly_receipts import Effect
from engine.v2.ops.nightly_session import (
    STATE_DIR,
    SessionIdentity,
    atomic_write,
    load_session,
)

__all__ = ["run_history_and_capture"]

_SCHEMA = "nightly_history_inputs.v1"
_FAMILIES = ("ledger/predictions", "ledger/outcomes")


def _pins(source_root: Path) -> tuple[tuple[str, str], ...]:
    if not (source_root / "ledger").is_dir():
        raise fail("SOURCE_NOT_FOUND", "source root has no ledger directory")
    found = []
    for relative in _FAMILIES:
        directory = source_root / relative
        if directory.is_symlink():
            raise fail("INPUT_CHANGED", "ledger history directory is a symlink")
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.is_symlink() or not path.is_file():
                raise fail("INPUT_CHANGED", "ledger history file is not a regular file",
                           details={"file": path.name})
            found.append((path.relative_to(source_root).as_posix(),
                           file_hash(path).removeprefix("sha256:")))
    return tuple(sorted(found))


def _pin_path(root: Path, run_id: str) -> Path:
    return Path(root).joinpath(*STATE_DIR, f"{run_id}.history-inputs.json")


def _read_pins(path, run_id, through):
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable") from exc
    if not isinstance(document, dict):
        raise fail("INTEGRITY_FAILED", "history file pins are unreadable")
    if (document.get("schema") != _SCHEMA or document.get("run_id") != run_id
            or document.get("through") != through
            or not isinstance(document.get("files"), list)):
        raise fail("INTEGRITY_FAILED", "history file pins failed identity checks")
    pins = tuple(tuple(pair) for pair in document["files"]
                 if isinstance(pair, list) and len(pair) == 2)
    if len(pins) != len(document["files"]):
        raise fail("INTEGRITY_FAILED", "history file pins are malformed")
    return pins


def _load_or_dry_run_pins(root, identity, generation, source_root, conn, clock):
    state = load_session(root, identity)
    if (state is None or type(generation) is not int or state.active.generation != generation
            or state.active.status != "started"):
        raise fail("CHECKPOINT_INCOMPATIBLE", "generation is not the active started one")
    through = (date.fromisoformat(identity.as_of) - timedelta(days=1)).isoformat()
    path = _pin_path(root, state.active.run_id)
    pins = _read_pins(path, state.active.run_id, through)
    is_new = pins is None
    if is_new:
        pins = _pins(Path(source_root))
    # Validate the same pinned set without committing; persist it only after the dry-run passes.
    import_history(conn, root, source_root, through=date.fromisoformat(through),
                   dry_run=True, clock=clock, pinned_files=pins)
    if is_new:
        doc = {"schema": _SCHEMA, "run_id": state.active.run_id,
               "through": through, "files": [list(pair) for pair in pins]}
        atomic_write(path, json.dumps(doc, sort_keys=True))
    return state.active.run_id, through, pins


def _artifact_successor(path: Path, digest: str, predecessor: str) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise fail("INPUT_CHANGED", "nightly artifact cannot be read") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise fail("INPUT_CHANGED", "nightly artifact differs from its pinned receipt")
    return predecessor


def run_history_and_capture(root: Path, identity: SessionIdentity, generation: int,
                            start_snapshot: str, *, conn, source_root, input_output,
                            tickers=(), context_tickers=(), year_start: int, year_end: int,
                            clock, kinds=capture_inputs.BARRIER_KINDS) -> ChainResult:
    """Dry-run and import pinned ledger files through D-1, then capture inputs.

    ``start_snapshot`` is the exact snapshot committed by the preceding
    price-history-capture step. This function does not advance the snapshot.
    """
    source_root = Path(source_root)
    run_id, through, pins = _load_or_dry_run_pins(
        root, identity, generation, source_root, conn, clock)
    reports = Path(root).joinpath(*STATE_DIR)
    history_path = reports / f"{run_id}.history-complete.json"
    history_doc = {"schema": "nightly_history_receipt.v1", "run_id": run_id,
                   "through": through, "files": [list(pair) for pair in pins]}
    history_text = json.dumps(history_doc, sort_keys=True)
    history_bytes = history_text.encode()
    history_hash = hashlib.sha256(history_bytes).hexdigest()

    def history_effect(_predecessor):
        return Effect("artifact", str(history_path), history_hash)

    def import_pinned_history(_predecessor):
        import_history(conn, root, source_root, through=date.fromisoformat(through),
                       dry_run=False, clock=clock, pinned_files=pins)
        atomic_write(history_path, history_text)

    history_step = ChainStep(
        "history_import", history_effect, import_pinned_history,
        lambda predecessor: _artifact_successor(history_path, history_hash, predecessor))

    output = Path(input_output).resolve()
    capture_holder = {}

    def capture_effect(_predecessor):
        manifest = capture_inputs.capture(
            source_root, as_of=identity.as_of, tickers=tickers,
            context_tickers=context_tickers, year_start=year_start, year_end=year_end,
            kinds=kinds)
        text = json.dumps(to_document(manifest), indent=2, sort_keys=True)
        data = text.encode()
        capture_holder.update(manifest=manifest, text=text,
                              digest=hashlib.sha256(data).hexdigest())
        return Effect("artifact", str(output), capture_holder["digest"])

    def write_input_manifest(_predecessor):
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(output, capture_holder["text"])

    input_step = ChainStep(
        "input_capture", capture_effect, write_input_manifest,
        lambda predecessor: _artifact_successor(
            output, capture_holder["digest"], predecessor))
    return run_snapshot_chain(root, identity, generation, start_snapshot,
                              [history_step, input_step], conn=conn)