from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from engine.dashboard import warm_serving
from engine.data.features import tier4


def test_needed_folds_uses_earlier_of_event_and_decision_fold():
    result = warm_serving.needed_folds(
        [pd.Timestamp("2026-10-25"), pd.Timestamp("2026-11-03")],
        "2026-10-20",
        "2026-10-20",
    )
    assert result == [pd.Timestamp("2026-10-01")]


def test_needed_folds_backfill_spans_decision_months():
    result = warm_serving.needed_folds(
        ["2026-11-03"],
        "2026-09-28",
        "2026-10-20",
    )
    assert result == [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01")]


def test_missing_jobs_skips_cached_files(monkeypatch, tmp_path):
    monkeypatch.setattr(tier4.store, "file_sha256", lambda p: "a" * 64)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    fold = pd.Timestamp("2026-10-01")
    jobs = warm_serving.missing_jobs([fold])
    assert len(jobs) == len(tier4.PRODUCES)
    assert all(fold_str == "2026-10-01" for _, fold_str in jobs)

    path = tier4._serving_path(tier4.feature_model("pred_abs_move").model_id, fold, "a" * 64)
    path.touch()

    jobs_after = warm_serving.missing_jobs([fold])
    assert "pred_abs_move" not in [produces for produces, _ in jobs_after]


def test_warm_no_missing_does_not_spawn(monkeypatch):
    monkeypatch.setattr(warm_serving, "missing_jobs", lambda folds: [])

    def boom(*args, **kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(warm_serving.subprocess, "run", boom)
    result = warm_serving.warm(
        ["2026-10-25"], "2026-10-20", "2026-10-20", log=lambda m: None
    )
    assert result["fitted"] == 0


def test_warm_subprocess_failure_raises(monkeypatch):
    monkeypatch.setattr(
        warm_serving, "missing_jobs", lambda folds: [("pred_abs_move", "2026-10-01")]
    )
    with pytest.raises(warm_serving.WarmError, match="exited 3"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "import sys; sys.exit(3)"],
            log=lambda m: None,
        )


def test_warm_exit_zero_but_still_missing_raises(monkeypatch):
    monkeypatch.setattr(
        warm_serving, "missing_jobs", lambda folds: [("pred_abs_move", "2026-10-01")]
    )
    with pytest.raises(warm_serving.WarmError, match="still missing"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "pass"],
            log=lambda m: None,
        )


def test_warm_success(monkeypatch):
    calls = {"n": 0}

    def fake_missing(folds):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("pred_abs_move", "2026-10-01")]
        return []

    monkeypatch.setattr(warm_serving, "missing_jobs", fake_missing)
    result = warm_serving.warm(
        ["2026-10-25"],
        "2026-10-20",
        "2026-10-20",
        command=[sys.executable, "-c", "pass"],
        log=lambda m: None,
    )
    assert result["fitted"] == 1


def test_nightly_wires_warm_failure_to_a_stop():
    source = Path(warm_serving.__file__).with_name("nightly.py").read_text()
    assert "warm_serving.warm(" in source
    assert "WARM_SERVING_FAILED" in source
