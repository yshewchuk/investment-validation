from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.models.training.fold_store import TrainingRefused
from tools import refresh_chooser_pool_candidates as refresh


def _rows(ids, markers=None, exit_dates=None):
    markers = markers or list(range(len(ids)))
    count = len(ids)
    return pd.DataFrame({
        "event_id": ids, "arm": ["N7q2_-1_-1_1"] * count,
        "variant": [0] * count, "fill_alpha": [0.5] * count,
        "ticker": ["XYZ"] * count, "event_date": pd.to_datetime(["2026-09-01"] * count),
        "entry_date": pd.to_datetime(["2026-09-01"] * count),
        "exit_date": pd.to_datetime(exit_dates or ["2026-09-30"] * count),
        "expiry": pd.to_datetime(["2026-10-02"] * count),
        "entry_cost": [1.0] * count, "exit_value": [3.0] * count,
        "exit_value_expiry": [3.0] * count, "exit_arb_ok": [True] * count,
        "spot_exit": [99.0] * count, "dte_exit": [3] * count,
        "spot_expiry": [101.0] * count, "pnl": [2.0] * count, "ret": [2.0] * count,
        "exp_pnl_sim": [1.0] * count, "width_over_forecast": [0.2] * count,
        "anchor_over_spot": [1.0] * count, "n_legs": [2] * count,
        "rel_spread": [0.1] * count, "session": ["AMC"] * count, "marker": markers,
    })


def test_month_requires_canonical_yyyy_mm():
    assert refresh.require_month("2026-10") == "2026-10"
    for invalid in ("2026-13", "2026-1", "2026-10x"):
        with pytest.raises(TrainingRefused, match="invalid_exit_month"):
            refresh.require_month(invalid)


def test_append_deduplicates_and_preserves_existing_rows_with_backup(tmp_path):
    path = tmp_path / "candidates.parquet"
    original = _rows(["old"], [1], ["2026-09-30"])
    original.loc[0, ["exit_arb_ok", "exit_value", "exit_value_expiry", "entry_cost", "pnl", "ret"]] = [
        False, -1.0, float("nan"), 2.0, -3.0, -1.5]
    original.to_parquet(path, index=False)
    fresh = _rows(["old", "new", "new"], [99, 2, 3],
                  ["2026-10-02", "2026-10-15", "2026-10-16"])
    fresh.loc[1, ["entry_cost", "exit_value", "exit_value_expiry", "pnl", "ret"]] = [2.0, 3.0, 3.0, 999.0, 999.0]
    fresh.loc[0, ["exit_arb_ok", "exit_value", "exit_value_expiry", "entry_cost",
                  "pnl", "ret", "spot_exit", "dte_exit", "spot_expiry"]] = [
        False, 3.0, 3.0, 1.0, 999.0, 999.0, 101.0, 0, 101.0]

    assert refresh.append_candidates(path, fresh) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["old", "marker"] == 1
    assert result.loc["old", "exit_date"] == pd.Timestamp("2026-09-30")
    assert result.loc["old", "spot_exit"] == 99.0
    assert result.loc["old", "dte_exit"] == 3
    assert result.loc["old", "exit_value"] == -1.0
    assert result.loc["old", "pnl"] == -3.0
    assert result.loc["old", "ret"] == -1.5
    assert result.loc["new", "marker"] == 2
    assert result.loc["new", "pnl"] == 1.0
    assert result.loc["new", "ret"] == 0.5
    assert list(pd.read_parquet(path).columns) == list(original.columns)
    from tools import build_chooser_pool
    built = build_chooser_pool.build(path).set_index("event_id")
    assert "old" not in built.index
    assert built.loc["new", "pnl"] == 1.0
    fresh.loc[0, ["pnl", "ret"]] = [2.0, 2.0]
    before_rerun = path.read_bytes()
    assert refresh.append_candidates(path, fresh) == 0
    assert path.read_bytes() == before_rerun
    assert len(result) == 2
    pd.testing.assert_frame_equal(pd.read_parquet(path.with_suffix(".parquet.bak")), original)


def test_refresh_refuses_month_already_present_before_builder(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    _rows(["existing"], exit_dates=["2026-10-15"]).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(refresh.importlib.util, "spec_from_file_location", lambda *_: pytest.fail("builder must not run"))

    with pytest.raises(TrainingRefused, match="month_already_present"):
        refresh.refresh("2026-10", path)
    assert path.read_bytes() == before


def test_append_schema_refusal_leaves_prior_table_unchanged(tmp_path):
    path = tmp_path / "candidates.parquet"
    original = _rows(["old"])
    original.to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises(TrainingRefused, match="candidate_schema_mismatch"):
        refresh.append_candidates(path, _rows(["new"]).drop(columns="event_date"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["event_date", "entry_date", "exit_date"])
def test_append_preserves_existing_rows_with_missing_dates(tmp_path, field):
    path = tmp_path / "candidates.parquet"
    old = _rows(["old"])
    old.loc[0, field] = pd.NaT
    old.to_parquet(path, index=False)

    assert refresh.append_candidates(path, _rows(["new"])) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert pd.isna(result.loc["old", field])
    assert "new" in result.index


def test_amc_entry_before_event_is_admitted(tmp_path):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    fresh = _rows(["new"])
    fresh.loc[0, "event_date"] = pd.Timestamp("2026-10-02")

    assert refresh.append_candidates(path, fresh) == 1


def test_append_refuses_bad_dates_without_rewrite(tmp_path):
    path = tmp_path / "candidates.parquet"
    original = _rows(["old"])
    original.to_parquet(path, index=False)
    before = path.read_bytes()
    fresh = _rows(["new"])
    fresh.loc[0, "event_date"] = pd.Timestamp("2026-08-31")
    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, fresh)
    assert path.read_bytes() == before


def test_missing_stored_table_is_typed_refusal(tmp_path):
    with pytest.raises(TrainingRefused, match="candidate_table_missing"):
        refresh.append_candidates(tmp_path / "missing.parquet", _rows(["new"]))


def test_refresh_calls_monthly_builder_and_consumes_rows_only(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    calls = []

    def build_all(**kwargs):
        calls.append(kwargs)
        return {"trades": _rows(["new"]), "tallies": object(), "events": object(), "meta": object()}

    builder = SimpleNamespace(build_all=build_all)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None), name="fake_builder")
    monkeypatch.setattr(refresh.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(refresh.importlib.util, "module_from_spec", lambda _spec: builder)

    assert refresh.refresh("2026-10", path) == 1
    assert calls == [{"force": True, "limit_exit_month": "2026-10"}]
    assert set(pd.read_parquet(path)["event_id"]) == {"old", "new"}
    assert path.with_suffix(".parquet.bak").exists()


def test_builder_failure_is_a_typed_refusal_and_does_not_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    before = path.read_bytes()

    def fail(**_kwargs):
        raise RuntimeError("required chain missing")

    builder = SimpleNamespace(build_all=fail)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None), name="fake_builder")
    monkeypatch.setattr(refresh.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(refresh.importlib.util, "module_from_spec", lambda _spec: builder)
    with pytest.raises(TrainingRefused, match="candidate_build_failed"):
        refresh.refresh("2026-10", path)
    assert path.read_bytes() == before


def test_chooser_source_builder_keeps_event_date(tmp_path, monkeypatch):
    from tools import build_chooser_pool

    monkeypatch.setattr(build_chooser_pool, "conditional_exit", lambda frame: frame.copy())
    row = {
        "arm": "N7q2_-1_-1_1", "event_id": "evt", "ticker": "XYZ",
        "event_date": pd.Timestamp("2026-09-01"), "entry_date": pd.Timestamp("2026-09-01"),
        "exit_date": pd.Timestamp("2026-10-01"), "fill_alpha": 0.5,
        "entry_cost": 2.0, "exit_value": 3.0, "exit_value_expiry": 3.0,
        "exit_arb_ok": True, "exp_pnl_sim": 1.0, "width_over_forecast": 0.2,
        "anchor_over_spot": 1.0, "n_legs": 2, "rel_spread": 0.1,
    }
    row.update({column: 0.0 for column in build_chooser_pool.Scorer._CHOOSER_ANALOG_DIMS})
    source = tmp_path / "candidates.parquet"
    pd.DataFrame([row]).to_parquet(source, index=False)

    result = build_chooser_pool.build(source)
    assert result.loc[0, "event_date"] == pd.Timestamp("2026-09-01")


def test_publication_fsyncs_staged_file_before_replace_and_directory_after(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    events = []
    real_fsync, real_replace = refresh.os.fsync, refresh.os.replace

    def record_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def record_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(refresh.os, "fsync", record_fsync)
    monkeypatch.setattr(refresh.os, "replace", record_replace)
    assert refresh.append_candidates(path, _rows(["new"])) == 1
    assert events == ["fsync", "replace", "fsync"]


def test_retained_repaired_rows_are_not_rewritten_by_month_refresh(tmp_path):
    path = tmp_path / "candidates.parquet"
    old = _rows(["historical"], exit_dates=["2026-09-30"])
    old.loc[0, ["exit_arb_ok", "exit_value"]] = [False, -1.0]
    old.to_parquet(path, index=False)

    assert refresh.append_candidates(path, _rows(["new"])) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["historical", "exit_date"] == pd.Timestamp("2026-09-30")
    assert result.loc["historical", "exit_value"] == -1.0


@pytest.mark.parametrize("field", ["exit_value_expiry", "entry_cost"])
def test_preserves_retained_repaired_row_missing_settlement_input(tmp_path, field):
    path = tmp_path / "candidates.parquet"
    old = _rows(["old"])
    old.loc[0, ["exit_arb_ok", "exit_value"]] = [False, -1.0]
    old.loc[0, field] = float("nan")
    old.to_parquet(path, index=False)

    assert refresh.append_candidates(path, _rows(["new"])) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert pd.isna(result.loc["old", field])
    assert result.loc["old", "exit_value"] == -1.0


@pytest.mark.parametrize("field", ["exit_value", "entry_cost"])
def test_refuses_fresh_non_repaired_row_missing_settlement_input(tmp_path, field):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    before = path.read_bytes()
    fresh = _rows(["new"])
    fresh.loc[0, field] = float("nan")

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, fresh)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["exit_value_expiry", "entry_cost"])
def test_refuses_fresh_repaired_row_missing_settlement_input(tmp_path, field):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    before = path.read_bytes()
    fresh = _rows(["new"])
    fresh.loc[0, ["exit_arb_ok", "exit_value"]] = [False, -1.0]
    fresh.loc[0, field] = float("nan")

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, fresh)
    assert path.read_bytes() == before


@pytest.mark.parametrize("expiry", [pd.NaT, pd.Timestamp("2026-08-31")])
def test_refuses_fresh_repaired_row_with_missing_or_pre_entry_expiry(tmp_path, expiry):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    before = path.read_bytes()
    fresh = _rows(["new"])
    fresh.loc[0, ["exit_arb_ok", "exit_value", "expiry"]] = [False, -1.0, expiry]

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, fresh)
    assert path.read_bytes() == before


def test_fresh_repaired_exit_date_uses_expiry(tmp_path):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    fresh = _rows(["new"], exit_dates=["2026-09-30"])
    fresh.loc[0, ["exit_arb_ok", "exit_value"]] = [False, -1.0]

    assert refresh.append_candidates(path, fresh) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["new", "exit_date"] == pd.Timestamp("2026-10-02")


def test_append_preserves_existing_repaired_row_with_pre_entry_expiry(tmp_path):
    path = tmp_path / "candidates.parquet"
    old = _rows(["old"])
    old.loc[0, ["exit_arb_ok", "exit_value", "expiry"]] = [False, -1.0, pd.Timestamp("2026-08-31")]
    old.to_parquet(path, index=False)

    assert refresh.append_candidates(path, _rows(["new"])) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["old", "exit_date"] == pd.Timestamp("2026-09-30")
    assert result.loc["old", "expiry"] == pd.Timestamp("2026-08-31")


@pytest.mark.parametrize("exit_date", ["NaT", "2026-08-31"])
def test_append_refuses_missing_or_pre_entry_exit_date(tmp_path, exit_date):
    path = tmp_path / "candidates.parquet"
    _rows(["old"]).to_parquet(path, index=False)
    before = path.read_bytes()

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, _rows(["new"], exit_dates=[exit_date]))
    assert path.read_bytes() == before
