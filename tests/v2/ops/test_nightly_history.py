"""Real catalog and synthetic source coverage for slice 5 history resume."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from engine.v2.ops import ledger_history_import as history_import
from engine.v2.ops import nightly_history
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_session import SessionIdentity, ensure_session, mark_started
from tests.ops_support import catalog

IDENTITY = SessionIdentity("2024-01-01", "shadow", "selection-1", "catalog-1")


def _prediction(row_id: str, as_of: str) -> dict:
    return {"row_id": row_id, "as_of": as_of, "ticker": "AAA", "strategy": "TWIN-P",
            "event_date": "2024-02-01", "event_id": "event-1",
            "decision_ts": as_of + "T20:00:00Z", "written_at": as_of + "T20:00:01Z",
            "snapshot_hash": "sha256:" + "a" * 64}


def _write_predictions(source: Path, name: str, *rows: dict) -> None:
    path = source / "ledger" / "predictions" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _outcome(row_id: str, resolved_at: str) -> dict:
    return {"row_id": row_id, "resolved_at": resolved_at, "ticker": "AAA",
            "strategy": "TWIN-P", "event_date": "2024-02-01",
            "status": "unresolvable", "reason": "test"}


def _write_outcomes(source: Path, name: str, *rows: dict) -> None:
    path = source / "ledger" / "outcomes" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _setup(tmp_path: Path):
    ops_root = tmp_path / "ops"
    ops_root.mkdir()
    conn, clock, _ = catalog(ops_root)
    ensure_session(ops_root, IDENTITY)
    mark_started(ops_root, IDENTITY, 1)
    source = tmp_path / "legacy"
    return ops_root, conn, clock, source


def _run(ops_root, conn, clock, source):
    return nightly_history.run_pinned_history_import(
        ops_root, IDENTITY, 1, "snapshot-after-price-history", conn=conn,
        source_root=source, clock=clock)


def test_d1_history_import_and_resume_after_source_drift(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("prior-row", "2023-12-31"),
                       _prediction("session-row", "2024-01-01"))

    first = _run(ops_root, conn, clock, source)
    assert first.final_snapshot == "snapshot-after-price-history"
    assert first.executed == ("history_import",)
    imported = conn.execute("SELECT decision_id FROM decisions ORDER BY decision_id").fetchall()
    assert [row[0] for row in imported] == ["prediction:prior-row"]
    resumed = _run(ops_root, conn, clock, source)
    assert resumed.executed == ()
    assert resumed.skipped == ("history_import",)

    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("later-source-row", "2023-12-30"))
    drifted = _run(ops_root, conn, clock, source)
    assert drifted.executed == ()
    assert drifted.skipped == ("history_import",)
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


def test_pending_import_refuses_added_or_changed_pinned_files(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("prior-row", "2023-12-31"))
    original_import = nightly_history.import_history

    def interrupt_before_import(conn_arg, root, source_root, *, dry_run=False, **kwargs):
        if dry_run:
            return original_import(conn_arg, root, source_root, dry_run=True, **kwargs)
        raise RuntimeError("simulated stop before the first file import")

    monkeypatch.setattr(nightly_history, "import_history", interrupt_before_import)
    with pytest.raises(RuntimeError, match="before the first file"):
        _run(ops_root, conn, clock, source)
    monkeypatch.setattr(nightly_history, "import_history", original_import)

    _write_predictions(source, "2024-01-01.jsonl",
                       _prediction("added-row", "2023-12-31"))
    with pytest.raises(OpsError) as added:
        _run(ops_root, conn, clock, source)
    assert added.value.code == "INPUT_CHANGED"
    (source / "ledger" / "predictions" / "2024-01-01.jsonl").unlink()
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("prior-row", "2023-12-31"),
                       _prediction("new-row", "2023-12-31"))

    with pytest.raises(OpsError) as changed:
        _run(ops_root, conn, clock, source)
    assert changed.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    conn.close()


def test_resume_reuses_pins_after_a_later_file_fails_after_prior_commit(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("first-row", "2023-12-30"))
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("second-row", "2023-12-31"))
    original_import = nightly_history.import_history
    original_file_import = history_import._import_one_file

    def fail_second_file(conn_arg, path, *args, **kwargs):
        if path.name == "2023-12-31.jsonl":
            raise RuntimeError("simulated process loss after the first file commit")
        return original_file_import(conn_arg, path, *args, **kwargs)

    def interrupted_import(conn_arg, root, source_root, *, dry_run=False, **kwargs):
        if dry_run:
            return original_import(conn_arg, root, source_root, dry_run=True, **kwargs)
        monkeypatch.setattr(history_import, "_import_one_file", fail_second_file)
        try:
            return original_import(conn_arg, root, source_root, dry_run=False, **kwargs)
        finally:
            monkeypatch.setattr(history_import, "_import_one_file", original_file_import)

    monkeypatch.setattr(nightly_history, "import_history", interrupted_import)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        _run(ops_root, conn, clock, source)
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1

    monkeypatch.setattr(nightly_history, "import_history", original_import)
    resumed = _run(ops_root, conn, clock, source)
    assert resumed.executed == ("history_import",)
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2
    conn.close()


def test_d1_history_import_includes_outcomes_and_excludes_session_date(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("row", "2023-12-31"))
    _write_outcomes(source, "2023-12-31.jsonl",
                    _outcome("row", "2023-12-31"),
                    _outcome("session-row", "2024-01-01"))

    result = _run(ops_root, conn, clock, source)
    assert result.executed == ("history_import",)
    assert conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE decision_id LIKE 'outcome:row:%'").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM decisions "
        "WHERE decision_id LIKE 'outcome:session-row:%'").fetchone()[0] == 0
    conn.close()


def test_symlinked_history_file_refuses_before_import(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    directory = source / "ledger" / "predictions"
    directory.mkdir(parents=True)
    (directory / "linked.jsonl").symlink_to(tmp_path / "missing.jsonl")

    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source)
    assert exc.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    conn.close()


def test_malformed_persisted_pin_hash_is_integrity_failure(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    state = nightly_history.load_session(ops_root, IDENTITY)
    path = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    path.write_text(json.dumps({
        "schema": "nightly_history_inputs.v1", "run_id": state.active.run_id,
        "through": "2023-12-31", "files": [["ledger/predictions/a.jsonl", "bad"]]}))

    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source)
    assert exc.value.code == "INTEGRITY_FAILED"
    conn.close()


def test_pinned_history_file_unavailable_refuses(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    target = source / "ledger" / "predictions" / "2023-12-31.jsonl"
    _write_predictions(source, target.name, _prediction("row", "2023-12-31"))
    pins = nightly_history._pins(source)
    original_read_bytes = Path.read_bytes

    def unavailable(path):
        if path == target:
            raise OSError("simulated unavailable pinned file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    with pytest.raises(OpsError) as exc:
        history_import.import_history(
            conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
            dry_run=False, clock=clock, pinned_files=pins)
    assert exc.value.code == "INPUT_CHANGED"
    conn.close()


def test_importer_rejects_invalid_pinned_hash_shape(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("row", "2023-12-31"))

    with pytest.raises(OpsError) as exc:
        history_import.import_history(
            conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
            dry_run=True, clock=clock,
            pinned_files=(("ledger/predictions/2023-12-31.jsonl", "bad"),))
    assert exc.value.code == "INVALID_REQUEST"
    conn.close()
