from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from engine.dashboard import warm_serving
from engine.data.features import tier4


def test_needed_folds_crosses_month_boundary():
    result = warm_serving.needed_folds(["2026-11-03"], "2026-10-20", "2026-10-20")
    assert result == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")]


def test_needed_folds_single_month():
    result = warm_serving.needed_folds(["2026-10-25"], "2026-10-20")
    assert result == [pd.Timestamp("2026-10-01")]


def test_needed_folds_early_entry_lookback():
    result = warm_serving.needed_folds(["2026-11-05"], "2026-11-02")
    assert result == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")]


def test_needed_folds_empty_events():
    assert warm_serving.needed_folds([], "2026-10-20") == []


def test_missing_jobs_ready_file_skips_producer(monkeypatch, tmp_path):
    monkeypatch.setattr(tier4.store, "file_sha256", lambda p: "a" * 64)
    monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path)

    fold = pd.Timestamp("2026-10-01")
    assert warm_serving.missing_jobs([fold]) == [
        (p, "2026-10-01") for p in tier4.PRODUCES
    ]

    m = tier4.feature_model("pred_abs_move")
    path = tier4._serving_path(m.model_id, fold, "a" * 64)
    joblib.dump(
        {
            "model_id": m.model_id,
            "fold_start": "2026-10-01",
            "tier3_snapshot": "a" * 64,
            "features": list(m.features),
            "pool_pred": [],
            "pool_res": [],
        },
        path,
    )
    others = [p for p in tier4.PRODUCES if p != "pred_abs_move"]
    assert warm_serving.missing_jobs([fold]) == [(p, "2026-10-01") for p in others]

    stored = joblib.load(path)
    del stored["pool_pred"]
    joblib.dump(stored, path)
    assert warm_serving.missing_jobs([fold]) == [
        (p, "2026-10-01") for p in tier4.PRODUCES
    ]

    path.write_bytes(b"not a joblib file")
    assert warm_serving.missing_jobs([fold]) == [
        (p, "2026-10-01") for p in tier4.PRODUCES
    ]


def test_warm_no_missing_does_not_spawn(monkeypatch):
    monkeypatch.setattr(warm_serving, "missing_jobs", lambda folds, snapshot=None: [])
    monkeypatch.setattr(warm_serving, "_panel_snapshot", lambda: "a" * 64)

    def boom(*args, **kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(warm_serving.subprocess, "run", boom)
    result = warm_serving.warm(
        ["2026-10-25"], "2026-10-20", "2026-10-20", log=lambda m: None
    )
    assert result["folds"] == 1
    assert result["fitted"] == 0
    assert "elapsed_s" in result


def test_warm_subprocess_failure_raises(monkeypatch):
    monkeypatch.setattr(
        warm_serving,
        "missing_jobs",
        lambda folds, snapshot=None: [("pred_abs_move", "2026-10-01")],
    )
    monkeypatch.setattr(warm_serving, "_panel_snapshot", lambda: "a" * 64)
    with pytest.raises(warm_serving.WarmError, match="exited 3"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "import sys; sys.exit(3)"],
            log=lambda m: None,
        )


def test_warm_timeout_raises(monkeypatch):
    monkeypatch.setattr(
        warm_serving,
        "missing_jobs",
        lambda folds, snapshot=None: [("pred_abs_move", "2026-10-01")],
    )
    monkeypatch.setattr(warm_serving, "_panel_snapshot", lambda: "a" * 64)
    with pytest.raises(warm_serving.WarmError, match="timed out"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=0.5,
            log=lambda m: None,
        )


def test_warm_exit_zero_but_still_missing_raises(monkeypatch):
    monkeypatch.setattr(
        warm_serving,
        "missing_jobs",
        lambda folds, snapshot=None: [("pred_abs_move", "2026-10-01")],
    )
    monkeypatch.setattr(warm_serving, "_panel_snapshot", lambda: "a" * 64)
    with pytest.raises(warm_serving.WarmError, match="still missing"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "pass"],
            log=lambda m: None,
        )


def test_warm_panel_changed_raises(monkeypatch):
    snaps = {"n": 0}

    def fake_snapshot():
        snaps["n"] += 1
        return "a" * 64 if snaps["n"] == 1 else "b" * 64

    calls = {"n": 0}

    def fake_missing(folds, snapshot=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("pred_abs_move", "2026-10-01")]
        return []

    monkeypatch.setattr(warm_serving, "_panel_snapshot", fake_snapshot)
    monkeypatch.setattr(warm_serving, "missing_jobs", fake_missing)
    with pytest.raises(warm_serving.WarmError, match="panel changed"):
        warm_serving.warm(
            ["2026-10-25"],
            "2026-10-20",
            "2026-10-20",
            command=[sys.executable, "-c", "pass"],
            log=lambda m: None,
        )


def test_warm_success(monkeypatch):
    calls = {"n": 0}

    def fake_missing(folds, snapshot=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("pred_abs_move", "2026-10-01")]
        return []

    monkeypatch.setattr(warm_serving, "missing_jobs", fake_missing)
    monkeypatch.setattr(warm_serving, "_panel_snapshot", lambda: "a" * 64)
    result = warm_serving.warm(
        ["2026-10-25"],
        "2026-10-20",
        "2026-10-20",
        command=[sys.executable, "-c", "pass"],
        log=lambda m: None,
    )
    assert result["folds"] == 1
    assert result["fitted"] == 1
    assert "elapsed_s" in result


def test_nightly_wires_warm_failure_to_a_stop():
    source = Path(warm_serving.__file__).with_name("nightly.py").read_text()
    assert "warm_serving.warm(" in source
    assert "WARM_SERVING_FAILED" in source
