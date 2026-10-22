from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.dashboard import nightly


def _warn(*args, **kwargs):
    raise AssertionError("scoring half must not run under fetch_only")


def _null_target(*args, **kwargs):
    return None


def _fake_refresh_half(monkeypatch, tmp_path, rebuild=None):
    def events_table(*args, **kwargs):
        return pd.DataFrame({
            "event_id": [1],
            "ticker": ["AAA"],
            "event_date": ["2026-10-25"],
            "session": ["amc"],
            "date_conflict": [False],
        })

    if rebuild is None:
        def rebuild(*args, **kwargs):
            return SimpleNamespace(elapsed_s=1.0, snapshot="abc", gap_fill={})

    def finalize(result):
        return SimpleNamespace(snapshot="abc")

    class FakeCal:
        def shift(self, ts, n):
            return ts

    monkeypatch.setattr("engine.data.store.read_table", events_table)
    monkeypatch.setattr(nightly, "refresh_calendar_data",
                        lambda *a, **k: {"ok": True})
    monkeypatch.setattr("engine.data.finality.resolve_final_session",
                        lambda *a, **k: SimpleNamespace(
                            date="2026-10-22", detail="",
                            as_dict=lambda: {"is_final": True,
                                             "date": "2026-10-22"}))
    monkeypatch.setattr("engine.calendar.trading_calendar",
                        lambda *a, **k: FakeCal())
    monkeypatch.setattr(nightly, "validate_refresh",
                        lambda *a, **k: [{"name": "x", "passed": True,
                                          "detail": ""}])
    monkeypatch.setattr("engine.data.pulls.computed_moves.read_state",
                        lambda *a, **k: {"moves_through": None})
    monkeypatch.setattr("engine.data.pulls.computed_moves.target_tickers",
                        lambda *a, **k: ([], {}))
    monkeypatch.setattr("engine.data.pulls.computed_moves."
                        "WATERMARK_OVERLAP_SESSIONS", 1, raising=False)
    monkeypatch.setattr("engine.data.rebuild.rebuild", rebuild)
    monkeypatch.setattr("engine.data.rebuild.finalize_rebuild", finalize)
    monkeypatch.setattr(nightly, "repair_tier4_gap_if_needed",
                        lambda *a, **k: None)
    monkeypatch.setattr(nightly, "_panel_staleness_flags", lambda *a, **k: [])
    monkeypatch.setattr(nightly, "_read_state", lambda *a, **k: {})
    flag_calls = []

    def write_flags(as_of, flags, steps):
        flag_calls.append(as_of)
        return tmp_path / "flags.json"

    monkeypatch.setattr(nightly, "_write_flag_report", write_flags)
    monkeypatch.setattr(nightly, "_write_state", _warn)
    monkeypatch.setattr("engine.score.Scorer", _warn)
    monkeypatch.setattr("engine.score.score_calendar", _warn)
    monkeypatch.setattr("engine.ledger.snapshot", _warn)
    monkeypatch.setattr("engine.dashboard.warm_serving.warm", _warn)
    monkeypatch.setattr("engine.dashboard.publish.publish_bundle", _warn)
    return flag_calls


def _assert_scoring_steps_skipped(report):
    for name in ("warm_serving", "score", "ledger", "settle", "ladder",
                 "backfill", "model_evidence", "render", "selfcheck",
                 "publish", "backup"):
        assert report.steps[name] == {"skipped": True,
                                      "reason": "fetch_only"}


@pytest.mark.parametrize("argv", [
    ["--fetch-only", "--no-tiers"],
    ["--fetch-only", "--no-refresh"],
    ["--fetch-only", "--backup"],
])
def test_cli_rejects_fetch_only_conflicts(monkeypatch, argv):
    monkeypatch.setattr(nightly, "single_run_lock",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", _warn)
    with pytest.raises(SystemExit) as exc:
        nightly.main(argv)
    assert exc.value.code == 2


def test_cli_passes_fetch_only_through(tmp_path, monkeypatch):
    calls = {}

    def record(*args, **kwargs):
        calls.update(kwargs)
        return nightly.NightlyReport(as_of="2026-10-22",
                                     requested_as_of="2026-10-22")

    monkeypatch.setattr(nightly.paths, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "single_run_lock",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(nightly, "run_nightly", record)

    assert nightly.main(["--as-of", "2026-10-22", "--fetch-only"]) == 0
    assert calls["fetch_only"] is True
    assert (tmp_path / "reports" / "nightly_2026-10-22.json").exists()


@pytest.mark.parametrize("kwargs", [
    {"fetch_only": True, "refresh": False},
    {"fetch_only": True, "tiers": False},
    {"fetch_only": True, "backup": True},
])
def test_run_nightly_rejects_fetch_only_conflicts(kwargs):
    with pytest.raises(ValueError, match="fetch_only"):
        nightly.run_nightly(**kwargs)


def test_fetch_only_happy_path_skips_scoring_half(tmp_path, monkeypatch):
    flag_calls = _fake_refresh_half(monkeypatch, tmp_path)
    report = nightly.run_nightly(
        "2026-10-22", fetch_only=True,
        bundle_dir=tmp_path / "b", target=tmp_path / "t",
        chain_sessions=1,
    )
    assert report.as_of == "2026-10-22"
    assert report.steps["tiers"]["degraded"] is False
    _assert_scoring_steps_skipped(report)
    assert len(flag_calls) == 1
    json.dumps(report.as_dict(), default=str)


def test_fetch_only_degrades_when_rebuild_fails(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    _fake_refresh_half(monkeypatch, tmp_path, rebuild=boom)

    def mark_degraded(report, exc):
        report.steps["tiers"] = {"degraded": True,
                                 "error": type(exc).__name__}

    monkeypatch.setattr(nightly, "mark_tiers_degraded", mark_degraded)
    report = nightly.run_nightly(
        "2026-10-22", fetch_only=True,
        bundle_dir=tmp_path / "b", target=tmp_path / "t",
        chain_sessions=1,
    )
    assert report.steps["tiers"]["degraded"] is True
    _assert_scoring_steps_skipped(report)