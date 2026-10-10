from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from engine.dashboard import nightly


def test_cli_writes_standard_report_without_json_flag(tmp_path, monkeypatch):
    requested_date = "2026-10-09"
    report = nightly.NightlyReport(
        as_of=requested_date,
        requested_as_of=None,
        resolved_as_of=requested_date,
        finality={"is_final": True},
        steps={"score": {"ok": True}},
        flags=[],
    )
    monkeypatch.setattr(nightly.paths, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "single_run_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", lambda *args, **kwargs: report)

    result = nightly.main([
        "--as-of", requested_date,
        "--no-refresh",
        "--no-tiers",
        "--no-publish",
        "--no-backfill",
    ])

    assert result == 0
    report_path = Path(tmp_path) / "reports" / f"nightly_{requested_date}.json"
    document = json.loads(report_path.read_text(encoding="utf-8"))
    assert {
        "as_of", "requested_as_of", "resolved_as_of", "finality", "steps",
        "timeline", "flags", "stopped", "elapsed_s",
    }.issubset(document)
    assert document["requested_as_of"] is None
    assert document["finality"] == {"is_final": True}


def test_cli_reports_standard_report_write_failure(tmp_path, monkeypatch, capsys):
    requested_date = "2026-10-09"
    report = nightly.NightlyReport(as_of=requested_date, requested_as_of=requested_date)
    monkeypatch.setattr(nightly.paths, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "single_run_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", lambda *args, **kwargs: report)

    def fail_write(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(nightly, "_write_report_atomic", fail_write)
    result = nightly.main(["--as-of", requested_date])

    assert result == 1
    assert "Failed to write standard nightly report: disk full" in capsys.readouterr().err


def test_cli_keeps_explicit_json_destination(tmp_path, monkeypatch):
    requested_date = "2026-10-09"
    report = nightly.NightlyReport(as_of=requested_date, requested_as_of=requested_date)
    monkeypatch.setattr(nightly.paths, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "single_run_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", lambda *args, **kwargs: report)
    explicit_path = tmp_path / "custom" / "report.json"
    explicit_path.parent.mkdir()

    result = nightly.main(["--as-of", requested_date, "--json", str(explicit_path)])

    assert result == 0
    assert json.loads(explicit_path.read_text(encoding="utf-8"))["requested_as_of"] == requested_date
    assert (tmp_path / "reports" / f"nightly_{requested_date}.json").is_file()

    monkeypatch.chdir(tmp_path)
    direct_writes = []
    original_write_text = Path.write_text

    def track_write_text(path, *args, **kwargs):
        direct_writes.append(path.resolve())
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", track_write_text)
    relative_standard = Path("reports") / f"nightly_{requested_date}.json"
    result = nightly.main(["--as-of", requested_date, "--json", str(relative_standard)])

    assert result == 0
    assert direct_writes == []


def test_atomic_report_writer_replaces_and_cleans_temporary_file(tmp_path):
    report_path = tmp_path / "reports" / "nightly_2026-10-09.json"
    report_path.parent.mkdir()
    report_path.write_text("old", encoding="utf-8")

    nightly._write_report_atomic(report_path, "new")

    assert report_path.read_text(encoding="utf-8") == "new"
    assert report_path.stat().st_mode & 0o777 == 0o644
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def test_atomic_report_writer_preserves_old_file_on_replace_failure(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "nightly_2026-10-09.json"
    report_path.parent.mkdir()
    report_path.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(nightly.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        nightly._write_report_atomic(report_path, "new")

    assert report_path.read_text(encoding="utf-8") == "old"
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def test_cli_reports_optional_json_write_failure_after_standard_report(tmp_path, monkeypatch, capsys):
    requested_date = "2026-10-09"
    report = nightly.NightlyReport(as_of=requested_date, requested_as_of=requested_date)
    monkeypatch.setattr(nightly.paths, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "single_run_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", lambda *args, **kwargs: report)

    result = nightly.main([
        "--as-of", requested_date,
        "--json", str(tmp_path / "reports"),
    ])

    assert result == 1
    assert (tmp_path / "reports" / f"nightly_{requested_date}.json").is_file()
    assert "Failed to write --json report:" in capsys.readouterr().err
