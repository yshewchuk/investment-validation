"""Real catalog and synthetic source coverage for slice 5 history resume."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.ops import ledger_history_import as history_import
from engine.v2.ops import nightly_history
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_session import SessionIdentity, ensure_session, mark_started
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_capture_inputs import _build_fixture

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


def _setup(tmp_path: Path):
    ops_root = tmp_path / "ops"
    ops_root.mkdir()
    conn, clock, _ = catalog(ops_root)
    ensure_session(ops_root, IDENTITY)
    mark_started(ops_root, IDENTITY, 1)
    source = tmp_path / "legacy"
    _build_fixture(source, year=2024, with_orats=False)
    output = tmp_path / "reports" / "nightly" / "input-manifest.json"
    return ops_root, conn, clock, source, output


def _run(ops_root, conn, clock, source, output):
    return nightly_history.run_history_and_capture(
        ops_root, IDENTITY, 1, "snapshot-after-price-history", conn=conn,
        source_root=source, input_output=output, tickers=("AAA",), context_tickers=("AAA",),
        year_start=2023, year_end=2024, clock=clock, kinds=("legacy_features",))


def test_d1_history_import_and_real_input_capture_resume(tmp_path):
    ops_root, conn, clock, source, output = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("prior-row", "2023-12-31"),
                       _prediction("session-row", "2024-01-01"))

    first = _run(ops_root, conn, clock, source, output)
    assert first.final_snapshot == "snapshot-after-price-history"
    assert first.executed == ("history_import", "input_capture")
    imported = conn.execute("SELECT decision_id FROM decisions ORDER BY decision_id").fetchall()
    assert [row[0] for row in imported] == ["prediction:prior-row"]
    document = json.loads(output.read_text())
    assert any(ref["path"].endswith("moves_AAA.json") for ref in document["file_refs"])

    first_bytes = output.read_bytes()
    resumed = _run(ops_root, conn, clock, source, output)
    assert resumed.executed == ()
    assert resumed.skipped == ("history_import", "input_capture")
    assert output.read_bytes() == first_bytes
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


def test_pinned_file_change_refuses_without_changing_imported_history(tmp_path):
    ops_root, conn, clock, source, output = _setup(tmp_path)
    _write_predictions(source, "2023-12-31.jsonl",
                              _prediction("prior-row", "2023-12-31"))
    _run(ops_root, conn, clock, source, output)
    _write_predictions(source, "2024-01-01.jsonl",
                       _prediction("added-row", "2023-12-31"))
    with pytest.raises(OpsError) as added:
        _run(ops_root, conn, clock, source, output)
    assert added.value.code == "INPUT_CHANGED"
    (source / "ledger" / "predictions" / "2024-01-01.jsonl").unlink()
    _write_predictions(source, "2023-12-31.jsonl",
                       _prediction("prior-row", "2023-12-31"),
                       _prediction("new-row", "2023-12-31"))

    with pytest.raises(OpsError) as exc:
        _run(ops_root, conn, clock, source, output)
    assert exc.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    conn.close()


def test_resume_reuses_pins_after_a_later_file_fails_after_prior_commit(tmp_path, monkeypatch):
    ops_root, conn, clock, source, output = _setup(tmp_path)
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
        _run(ops_root, conn, clock, source, output)
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1

    monkeypatch.setattr(nightly_history, "import_history", original_import)
    resumed = _run(ops_root, conn, clock, source, output)
    assert resumed.executed == ("history_import", "input_capture")
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2
    conn.close()