"""P6-2 snapshot-mode numeric comparison of the rebuilt features tables.

Covers ``engine/v2/ops/features_compare.py`` and the snapshot half of
``legacy_adapter._check_features_current``/``_action_features``: a rebuilt
panel/tier4 that differs from the pinned materialization only by float noise
within tolerance still ``match``es (and records the size of the difference),
while a structural change, an out-of-tolerance float, a swapped pinned table
or a missing verdict all refuse with a typed ``FEATURES_STALE`` problem.

Every frame here is synthetic and tiny; ``engine.data.rebuild``'s builders are
monkeypatched to no-ops so no real Tier-3/Tier-4 rebuild ever runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.features_compare import IDENTICAL, compare_tables
from engine.v2.ops.legacy_adapter import _action_features, _check_features_current
from engine.v2.ops.snapshot_stages import launch_mode


def _frame(n=6):
    base = pd.Timestamp("2024-01-01")
    dates = pd.to_datetime([base + pd.Timedelta(days=i) for i in range(n)])
    values = [0.5 + 0.01 * (i + 1) for i in range(n)]
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "event_date": dates,
        "date": dates,
        "n_obs": pd.Series([i + 1 for i in range(n)], dtype="int64"),
        "label": [f"L{i}" for i in range(n)],
        "pred_abs_move": values,
        "pred_abs_move_p10": values,
        "pred_abs_move_p90": values,
        "pred_abs_move_sd": values,
    })


def _write(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _ulp(df, col, rows=(1, 3)):
    out = df.copy()
    for i in rows:
        out.loc[i, col] = np.nextafter(out.loc[i, col], np.inf)
    return out


def test_identical_tables_match(tmp_path):
    df = _frame()
    rebuilt, pinned = tmp_path / "rebuilt.parquet", tmp_path / "pinned.parquet"
    _write(rebuilt, df)
    _write(pinned, df)

    result = compare_tables("tier4", rebuilt, pinned)
    assert result["verdict"] == "match"
    assert result["max_abs_diff"] == 0.0
    assert result["mismatches"] == []


def test_one_ulp_float_noise_matches_and_is_recorded(tmp_path):
    df = _frame()
    rebuilt, pinned = tmp_path / "rebuilt.parquet", tmp_path / "pinned.parquet"
    _write(rebuilt, _ulp(df, "pred_abs_move"))
    _write(pinned, df)

    result = compare_tables("tier4", rebuilt, pinned)
    assert result["verdict"] == "match"
    assert 0 < result["max_abs_diff"] < 1e-12
    assert result["n_columns_differing"] == 1
    assert result["mismatches"] == []


def test_float_beyond_tolerance_refuses_with_details(tmp_path):
    df = _frame()
    rebuilt = df.copy()
    rebuilt.loc[2, "pred_abs_move_sd"] = rebuilt.loc[2, "pred_abs_move_sd"] + 1e-6
    rebuilt_path, pinned_path = tmp_path / "rebuilt.parquet", tmp_path / "pinned.parquet"
    _write(rebuilt_path, rebuilt)
    _write(pinned_path, df)

    result = compare_tables("tier4", rebuilt_path, pinned_path)
    assert result["verdict"] == "mismatch"
    first = result["mismatches"][0]
    assert first["reason"] == "float_beyond_tolerance"
    assert first["column"] == "pred_abs_move_sd"
    assert first["n_rows"] == 1
    assert first["max_abs_diff"] == pytest.approx(1e-6, rel=1e-3)


def _structural_rebuilt(case, df):
    if case == "key":
        out = df.copy()
        out.loc[2, "event_date"] = out.loc[2, "event_date"] + pd.Timedelta(days=1)
        return out
    if case == "row_count":
        return df.iloc[:-1].reset_index(drop=True)
    if case == "schema_rename":
        return df.rename(columns={"pred_abs_move_sd": "pred_sd"})
    if case == "dtype":
        out = df.copy()
        out["pred_abs_move_sd"] = out["pred_abs_move_sd"].astype("float32")
        return out
    if case == "non_float":
        out = df.copy()
        out.loc[2, "n_obs"] = out.loc[2, "n_obs"] + 1
        return out
    if case == "nan_pattern":
        out = df.copy()
        out.loc[2, "pred_abs_move"] = np.nan
        return out
    raise AssertionError(f"unknown structural case {case!r}")


@pytest.mark.parametrize("case,reason,column", [
    ("key", "key", "event_date"),
    ("row_count", "row_count", None),
    ("schema_rename", "schema", None),
    ("dtype", "schema", None),
    ("non_float", "non_float", "n_obs"),
    ("nan_pattern", "nan_pattern", "pred_abs_move"),
    ("table_missing", "table_missing", None),
], ids=["key", "row_count", "schema_rename", "dtype", "non_float", "nan_pattern",
        "table_missing"])
def test_structural_or_exact_changes_refuse(tmp_path, case, reason, column):
    df = _frame()
    rebuilt, pinned = tmp_path / "rebuilt.parquet", tmp_path / "pinned.parquet"
    if case == "table_missing":
        _write(rebuilt, df)
    else:
        _write(rebuilt, _structural_rebuilt(case, df))
        _write(pinned, df)

    result = compare_tables("tier4", rebuilt, pinned)
    assert result["verdict"] == "mismatch"
    assert result["mismatches"][0]["reason"] == reason
    if column is not None:
        assert result["mismatches"][0]["column"] == column


def test_legacy_mode_stays_byte_exact(monkeypatch, tmp_path):
    from engine import paths as paths_module
    from engine.data import store

    panel = tmp_path / "data" / "panel.parquet"
    tier4 = tmp_path / "data" / "tier4_forecasts.parquet"
    monkeypatch.setattr(paths_module, "PANEL", panel)
    monkeypatch.setattr(paths_module, "TIER4", tier4)
    df = _frame()
    _write(panel, df)
    _write(tier4, df)

    root = tmp_path / "session"
    root.mkdir()
    (root / "features.json").write_text(json.dumps({
        "panel_sha256": store.file_sha256(panel),
        "tier4_sha256": store.file_sha256(tier4),
    }))

    _write(tier4, _ulp(df, "pred_abs_move"))

    for parameters in (None, {"input_mode": "legacy"}):
        with pytest.raises(OpsError) as excinfo:
            _check_features_current(root, parameters)
        assert excinfo.value.code == "FEATURES_STALE"
        assert set(excinfo.value.problem.details["mismatches"]) == {"tier4_sha256"}


@pytest.fixture
def snapshot_case(tmp_path, monkeypatch):
    """A staging root whose rebuilt tier4 differs from the pinned tables only
    by 1-ulp float noise (or, with ``perturb=True``, by an out-of-tolerance
    float change), plus the pinned materialization root it was compared to."""
    from engine import paths as paths_module
    from engine.data import rebuild

    def build(perturb=False):
        st = tmp_path / "st"
        mat = tmp_path / "mat"
        st_features = st / "data" / "features"
        mat_features = mat / "data" / "features"
        base = _frame()

        _write(st_features / "panel.parquet", base)
        _write(mat_features / "panel.parquet", base)
        _write(mat_features / "tier4_forecasts.parquet", base)
        if perturb:
            rebuilt = base.copy()
            rebuilt.loc[2, "pred_abs_move_sd"] = rebuilt.loc[2, "pred_abs_move_sd"] + 1e-6
        else:
            rebuilt = _ulp(base, "pred_abs_move")
        _write(st_features / "tier4_forecasts.parquet", rebuilt)

        monkeypatch.setattr(paths_module, "ROOT", st)
        monkeypatch.setattr(paths_module, "PANEL", st_features / "panel.parquet")
        monkeypatch.setattr(paths_module, "TIER4", st_features / "tier4_forecasts.parquet")
        monkeypatch.setattr(rebuild, "build_panel_table", lambda: {"rows": 6})
        monkeypatch.setattr(rebuild, "build_tier4_table", lambda since=None: {"rows": 6})

        _action_features({}, st, cross_check={"materialization_root": str(mat)})
        return st, mat, paths_module

    return build


def test_snapshot_features_records_numeric_verdict(snapshot_case):
    from engine.data import store

    st, mat, _paths = snapshot_case()
    receipt = json.loads((st / "features.json").read_text())

    comparison = receipt["comparison"]
    assert comparison["id"] == "numeric.v1"
    assert receipt["verdict"] == "match"
    assert receipt["pinned_panel_sha256"] == \
        store.file_sha256(mat / "data" / "features" / "panel.parquet")

    tier4 = receipt["tables"]["tier4"]
    assert 0 < tier4["max_abs_diff"] < 1e-12
    assert receipt["tables"]["panel"] == dict(IDENTICAL, mismatches=[])


def test_snapshot_score_check_passes_against_the_pinned_tables(snapshot_case, monkeypatch):
    st, mat, paths = snapshot_case()
    features = mat / "data" / "features"
    monkeypatch.setattr(paths, "PANEL", features / "panel.parquet")
    monkeypatch.setattr(paths, "TIER4", features / "tier4_forecasts.parquet")

    assert _check_features_current(st, {"input_mode": "snapshot"}) is None

    with pytest.raises(OpsError) as excinfo:
        _check_features_current(st)
    assert excinfo.value.code == "FEATURES_STALE"
    assert "tier4_sha256" in excinfo.value.problem.details["mismatches"]


def test_table_swapped_after_the_features_stage_is_refused(snapshot_case, monkeypatch):
    st, mat, paths = snapshot_case()
    features = mat / "data" / "features"
    monkeypatch.setattr(paths, "PANEL", features / "panel.parquet")
    monkeypatch.setattr(paths, "TIER4", features / "tier4_forecasts.parquet")

    _write(features / "tier4_forecasts.parquet", _ulp(_frame(), "pred_abs_move_p90"))

    with pytest.raises(OpsError) as excinfo:
        _check_features_current(st, {"input_mode": "snapshot"})
    assert excinfo.value.code == "FEATURES_STALE"
    details = excinfo.value.problem.details
    assert details["reason"] == "pinned_changed"
    assert "tier4_sha256" in details["mismatches"]


def test_snapshot_receipt_without_a_verdict_is_refused(monkeypatch, tmp_path):
    from engine import paths as paths_module

    monkeypatch.setattr(paths_module, "PANEL", tmp_path / "panel.parquet")
    monkeypatch.setattr(paths_module, "TIER4", tmp_path / "tier4.parquet")

    root = tmp_path / "root"
    root.mkdir()
    (root / "features.json").write_text(
        json.dumps({"panel_sha256": "x", "tier4_sha256": "y"}))

    with pytest.raises(OpsError) as excinfo:
        _check_features_current(root, {"input_mode": "snapshot"})
    assert excinfo.value.code == "FEATURES_STALE"
    assert excinfo.value.problem.details["reason"] == "not_compared"


def test_snapshot_mismatch_verdict_is_refused_with_table_details(snapshot_case):
    st, _mat, _paths = snapshot_case(perturb=True)

    with pytest.raises(OpsError) as excinfo:
        _check_features_current(st, {"input_mode": "snapshot"})
    assert excinfo.value.code == "FEATURES_STALE"
    details = excinfo.value.problem.details
    assert details["reason"] == "mismatch"
    assert details["tables"]["tier4"]["mismatches"][0]["column"] == "pred_abs_move_sd"


def test_failure_details_reach_the_diagnostic_file_not_failure_json(snapshot_case, tmp_path):
    from engine.v2.ops import worker

    st, _mat, _paths = snapshot_case(perturb=True)
    with pytest.raises(OpsError) as excinfo:
        _check_features_current(st, {"input_mode": "snapshot"})

    diag_root = tmp_path / "failure"
    diag_root.mkdir()
    result = worker._failure_result(diag_root, excinfo.value)

    assert result["failure"] == "FEATURES_STALE"
    assert "details" not in result["problem"]
    details = json.loads((diag_root / "diagnostics" / "failure_details.json").read_text())
    assert details["tables"]["tier4"]["mismatches"][0]["reason"] == "float_beyond_tolerance"


def test_features_stage_is_a_cross_check_stage_and_launches_as_finality_check():
    from engine.v2.ops import nightly

    bindings = {name: "job#legacy_features" for name in (
        "legacy_manifest.json", "snapshot_ref.json",
        "materialization_request.json", "materialization_manifest.json")}
    spec = SimpleNamespace(kind="legacy_features", parameters={"input_bindings": bindings})
    assert launch_mode(spec) == "finality_check"

    legacy_spec = SimpleNamespace(
        kind="legacy_features",
        parameters={"input_bindings": {"legacy_manifest.json": "job#legacy_features"}})
    assert launch_mode(legacy_spec) == "legacy"

    assert "features" in nightly.CROSS_CHECK_STAGES
