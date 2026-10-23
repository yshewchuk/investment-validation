"""Real catalog and synthetic source coverage for slice 5 history resume."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from engine.v2.ops import ledger_history_import as history_import
from engine.v2.ops import nightly_history
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_receipts import Effect, begin_step, complete_step
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


def test_corrupt_history_completion_artifact_refuses_resume(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl", _prediction("row", "2023-12-31"))
    _run(ops_root, conn, clock, source)
    state = nightly_history.load_session(ops_root, IDENTITY)
    artifact = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-complete.json")
    artifact.write_text("corrupt")

    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source)
    assert exc.value.code == "INTEGRITY_FAILED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


def test_pinned_import_revalidates_added_file_after_per_file_commit(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl", _prediction("first", "2023-12-31"))
    original = history_import._import_one_file
    added = False

    def add_file_after_import(*args, **kwargs):
        nonlocal added
        result = original(*args, **kwargs)
        if not kwargs.get("dry_run") and not added:
            added = True
            _write_predictions(source, "late.jsonl", _prediction("late", "2023-12-31"))
        return result

    monkeypatch.setattr(history_import, "_import_one_file", add_file_after_import)
    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source)
    assert exc.value.code == "INPUT_CHANGED"
    assert [row[0] for row in conn.execute(
        "SELECT decision_id FROM decisions").fetchall()] == ["prediction:first"]
    state = nightly_history.load_session(ops_root, IDENTITY)
    artifact = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-complete.json")
    assert not artifact.exists()
    conn.close()


def test_history_pin_capture_rejects_symlinked_ledger_ancestor(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    outside = tmp_path / "outside-ledger"
    _write_predictions(outside, "2023-12-31.jsonl", _prediction("row", "2023-12-31"))
    source.mkdir()
    (source / "ledger").symlink_to(outside / "ledger", target_is_directory=True)

    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source)
    assert exc.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    conn.close()


def test_pinned_fifo_refuses_without_waiting_for_writer(tmp_path):
    source = tmp_path / "legacy"
    fifo = source / "ledger" / "predictions" / "blocked.jsonl"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)

    with pytest.raises(OpsError) as exc:
        history_import._read_pinned_bytes(source, fifo)
    assert exc.value.code == "INPUT_CHANGED"


def test_pinned_import_rejects_ledger_swap_during_final_revalidation(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    target = source / "ledger" / "predictions" / "2023-12-31.jsonl"
    _write_predictions(source, target.name, _prediction("row", "2023-12-31"))
    outside = tmp_path / "outside"
    _write_predictions(outside, target.name, _prediction("row", "2023-12-31"))
    pins = nightly_history._pins(source)
    original_open = os.open
    opens = 0

    def swap_on_second_open(path, flags, *args, **kwargs):
        nonlocal opens
        if Path(path) == target:
            opens += 1
            if opens == 2:
                (source / "ledger").rename(source / "ledger-original")
                (source / "ledger").symlink_to(outside / "ledger", target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_on_second_open)
    with pytest.raises(OpsError) as exc:
        history_import.import_history(
            conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
            dry_run=False, clock=clock, pinned_files=pins)
    assert exc.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


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


def test_changed_later_file_refuses_after_prior_file_commit(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("first-row", "2023-12-30"))
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("second-row", "2023-12-31"))
    original_import = nightly_history.import_history
    original_file_import = history_import._import_one_file

    def drift_second_file(conn_arg, path, *args, **kwargs):
        if path.name == "2023-12-31.jsonl":
            _write_predictions(source, "2023-12-31.jsonl",
                               _prediction("changed-second-row", "2023-12-31"))
        return original_file_import(conn_arg, path, *args, **kwargs)

    def drift_during_import(conn_arg, root, source_root, *, dry_run=False, **kwargs):
        if dry_run:
            return original_import(conn_arg, root, source_root, dry_run=True, **kwargs)
        monkeypatch.setattr(history_import, "_import_one_file", drift_second_file)
        try:
            return original_import(conn_arg, root, source_root, dry_run=False, **kwargs)
        finally:
            monkeypatch.setattr(history_import, "_import_one_file", original_file_import)

    monkeypatch.setattr(nightly_history, "import_history", drift_during_import)
    with pytest.raises(OpsError) as changed:
        _run(ops_root, conn, clock, source)
    assert changed.value.code == "INPUT_CHANGED"
    assert [row[0] for row in conn.execute(
        "SELECT decision_id FROM decisions").fetchall()] == ["prediction:first-row"]
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


def test_unpinned_history_import_keeps_glob_behavior_for_symlinked_file(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    target = tmp_path / "prediction.jsonl"
    target.write_text(json.dumps(_prediction("linked-row", "2023-12-31")) + "\n")
    link = source / "ledger" / "predictions" / "2023-12-31.jsonl"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    summary = history_import.import_history(
        conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
        dry_run=True, clock=clock)
    assert summary["families"]["predictions"]["files"] == 1
    assert summary["families"]["predictions"]["lines"] == 1
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
    original_open = os.open

    def unavailable(path, flags, *args, **kwargs):
        if Path(path) == target:
            raise OSError("simulated unavailable pinned file")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", unavailable)
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


def test_importer_accepts_list_pairs_for_pinned_files(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    path = source / "ledger" / "predictions" / "2023-12-31.jsonl"
    _write_predictions(source, path.name, _prediction("row", "2023-12-31"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    summary = history_import.import_history(
        conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
        dry_run=True, clock=clock,
        pinned_files=[["ledger/predictions/2023-12-31.jsonl", digest]])
    assert summary["dry_run"] is True
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    conn.close()


def test_overlapping_same_run_calls_keep_pins_stable_and_resume(tmp_path, monkeypatch):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl", _prediction("row", "2023-12-31"))
    first_pins_entered = Event()
    second_lock_attempted = Event()
    release_first = Event()
    guard = Lock()
    pins_calls = 0
    lock_calls = 0
    original_pins = nightly_history._pins
    original_lock = nightly_history._history_lock

    def paused_pins(source_root):
        nonlocal pins_calls
        with guard:
            pins_calls += 1
            first_call = pins_calls == 1
        if first_call:
            first_pins_entered.set()
            assert release_first.wait(5)
        return original_pins(source_root)

    def observed_lock(root, run_id):
        nonlocal lock_calls
        with guard:
            lock_calls += 1
            if lock_calls == 2:
                second_lock_attempted.set()
        return original_lock(root, run_id)

    monkeypatch.setattr(nightly_history, "_pins", paused_pins)
    monkeypatch.setattr(nightly_history, "_history_lock", observed_lock)
    start = Barrier(2)

    def invoke():
        thread_conn = open_catalog(ops_root / "ops.sqlite", clock=clock)
        try:
            start.wait(5)
            return _run(ops_root, thread_conn, clock, source)
        finally:
            thread_conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(invoke), pool.submit(invoke)]
        assert first_pins_entered.wait(5)
        assert second_lock_attempted.wait(5)
        release_first.set()
        results = [call.result(timeout=10) for call in calls]

    assert pins_calls == 1
    assert sorted((result.executed, result.skipped) for result in results) == sorted([
        (("history_import",), ()), ((), ("history_import",))])
    state = nightly_history.load_session(ops_root, IDENTITY)
    pin_path = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    persisted = json.loads(pin_path.read_text())
    assert persisted["files"] == [[
        "ledger/predictions/2023-12-31.jsonl",
        hashlib.sha256((source / "ledger/predictions/2023-12-31.jsonl").read_bytes()).hexdigest(),
    ]]
    resumed = _run(ops_root, conn, clock, source)
    assert resumed.executed == ()
    assert resumed.skipped == ("history_import",)
    assert json.loads(pin_path.read_text()) == persisted
    conn.close()


def test_import_history_reports_exact_through_coverage_and_keeps_older_rows(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("older-row", "2023-12-30"))
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("through-row-a", "2023-12-31"),
                       _prediction("through-row-b", "2023-12-31"))
    _write_outcomes(source, "2023-12-30.jsonl",
                    _outcome("older-outcome", "2023-12-30"))

    dry = history_import.import_history(
        conn, ops_root, source, through=date.fromisoformat("2023-12-31"),
        dry_run=True, clock=clock)
    assert dry["coverage"] == {
        "through": "2023-12-31",
        "families": {
            "predictions": {"rows": 2, "status": "present"},
            "outcomes": {"rows": 0, "status": "no_rows"},
        },
    }

    real = history_import.import_history(
        conn, ops_root, source, through=date.fromisoformat("2023-12-31"), clock=clock)
    assert real["coverage"] == dry["coverage"]
    imported = [row[0] for row in conn.execute(
        "SELECT decision_id FROM decisions ORDER BY decision_id").fetchall()]
    assert imported[1:] == [
        "prediction:older-row", "prediction:through-row-a", "prediction:through-row-b"]
    assert imported[0].startswith("outcome:older-outcome:")
    conn.close()


def test_pinned_history_receipt_carries_dry_run_coverage_and_stays_idempotent(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("older-row", "2023-12-30"))
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("through-row", "2023-12-31"))
    expected = {
        "through": "2023-12-31",
        "families": {
            "predictions": {"rows": 1, "status": "present"},
            "outcomes": {"rows": 0, "status": "no_rows"},
        },
    }

    _run(ops_root, conn, clock, source)
    state = nightly_history.load_session(ops_root, IDENTITY)
    inputs_path = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    inputs = json.loads(inputs_path.read_text())
    assert inputs["schema"] == "nightly_history_inputs.v1.1"
    assert inputs["coverage"] == expected

    receipt_path = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-complete.json")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["schema"] == "nightly_history_receipt.v1.1"
    assert receipt["coverage"] == expected

    resumed = _run(ops_root, conn, clock, source)
    assert resumed.skipped == ("history_import",)
    assert receipt_path.read_bytes() == receipt_bytes
    conn.close()


def test_legacy_pin_document_without_coverage_is_upgraded_from_pinned_dry_run(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("through-row", "2023-12-31"))
    state = nightly_history.load_session(ops_root, IDENTITY)
    path = ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-inputs.json")
    path.write_text(json.dumps({
        "schema": "nightly_history_inputs.v1", "run_id": state.active.run_id,
        "through": "2023-12-31",
        "files": [list(pair) for pair in nightly_history._pins(source)]}))

    result = _run(ops_root, conn, clock, source)
    assert result.executed == ("history_import",)
    upgraded = json.loads(path.read_text())
    assert upgraded["schema"] == "nightly_history_inputs.v1.1"
    assert upgraded["coverage"]["families"]["predictions"] == {
        "rows": 1, "status": "present"}
    receipt = json.loads(ops_root.joinpath(
        *nightly_history.STATE_DIR, f"{state.active.run_id}.history-complete.json").read_text())
    assert receipt["coverage"] == upgraded["coverage"]
    conn.close()


def test_both_families_can_report_no_rows_without_blocking_older_rows(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-30.jsonl",
                       _prediction("older-prediction", "2023-12-30"))
    _write_outcomes(source, "2023-12-30.jsonl", _outcome("older-outcome", "2023-12-30"))

    result = history_import.import_history(
        conn, ops_root, source, through=date.fromisoformat("2023-12-31"), clock=clock)
    assert result["coverage"] == {
        "through": "2023-12-31",
        "families": {
            "predictions": {"rows": 0, "status": "no_rows"},
            "outcomes": {"rows": 0, "status": "no_rows"},
        },
    }
    decision_ids = [row[0] for row in conn.execute(
        "SELECT decision_id FROM decisions ORDER BY decision_id").fetchall()]
    assert decision_ids[1:] == ["prediction:older-prediction"]
    assert decision_ids[0].startswith("outcome:older-outcome:")
    conn.close()


def test_completed_legacy_receipt_is_reused_without_reimport(tmp_path):
    ops_root, conn, clock, source = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl", _prediction("row", "2023-12-31"))
    state = nightly_history.load_session(ops_root, IDENTITY)
    state_dir = ops_root.joinpath(*nightly_history.STATE_DIR)
    artifact = state_dir / f"{state.active.run_id}.history-complete.json"
    pins = nightly_history._pins(source)
    legacy_text = json.dumps({
        "schema": "nightly_history_receipt.v1", "run_id": state.active.run_id,
        "through": "2023-12-31", "files": [list(pair) for pair in pins]}, sort_keys=True)
    artifact.write_text(legacy_text)
    (state_dir / f"{state.active.run_id}.history-inputs.json").write_text(json.dumps({
        "schema": "nightly_history_inputs.v1", "run_id": state.active.run_id,
        "through": "2023-12-31", "files": [list(pair) for pair in pins]}))

    digest = hashlib.sha256(
        b"history_import\0snapshot-after-price-history").hexdigest()
    effect = Effect("artifact", str(artifact),
                    hashlib.sha256(legacy_text.encode()).hexdigest())
    begin_step(ops_root, IDENTITY, 1, "history_import", digest, effect)
    complete_step(ops_root, IDENTITY, 1, "history_import", conn=conn)

    result = _run(ops_root, conn, clock, source)
    assert result.executed == ()
    assert result.skipped == ("history_import",)
    assert artifact.read_text() == legacy_text
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    inputs = json.loads((state_dir / f"{state.active.run_id}.history-inputs.json").read_text())
    assert inputs["schema"] == "nightly_history_inputs.v1.1"
    assert inputs["coverage"]["families"]["predictions"] == {
        "rows": 1, "status": "present"}
    conn.close()
