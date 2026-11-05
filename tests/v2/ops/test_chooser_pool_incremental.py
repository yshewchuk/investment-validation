import pandas as pd
from types import SimpleNamespace
import pytest

from engine.v2.models.training.fold_store import TrainingRefused
from tools import refresh_chooser_pool_candidates as refresh


def _row(event_id, value, event_date="2026-09-01", entry_date="2026-09-01", session="AMC"):
    return {
        "event_id": event_id, "arm": "N7q2_-1_-1_1", "variant": 0,
        "fill_alpha": 0.5, "event_date": pd.Timestamp(event_date),
        "entry_date": pd.Timestamp(entry_date), "exit_date": pd.Timestamp("2026-10-15"),
        "session": session, "marker": value,
    }


def test_month_requires_canonical_value():
    assert refresh.require_month("2026-10") == "2026-10"
    with pytest.raises(TrainingRefused, match="invalid_exit_month"):
        refresh.require_month("2026-13")


def test_append_keeps_existing_key_and_adds_only_unseen_rows(tmp_path):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)
    fresh = pd.DataFrame([_row("existing", 99), _row("new", 2), _row("new", 3)])

    assert refresh.append_candidates(path, fresh) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["existing", "marker"] == 1
    assert result.loc["new", "marker"] == 2
    assert len(result) == 2


def test_append_refuses_future_feature_date_without_changing_table(tmp_path):
    path = tmp_path / "candidates.parquet"
    original = pd.DataFrame([_row("existing", 1)])
    original.to_parquet(path, index=False)

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(
            path, pd.DataFrame([_row("new", 2, event_date="2026-10-02", entry_date="2026-10-01")])
        )

    pd.testing.assert_frame_equal(pd.read_parquet(path), original)


def test_append_accepts_bmo_event_date_after_entry_date(tmp_path):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)

    fresh = pd.DataFrame([_row(
        "bmo", 2, event_date="2026-10-02", entry_date="2026-10-01", session="BMO")])
    assert refresh.append_candidates(path, fresh) == 1
    assert "bmo" in set(pd.read_parquet(path)["event_id"])


def test_atomic_append_failure_preserves_table_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    original = pd.DataFrame([_row("existing", 1)])
    original.to_parquet(path, index=False)

    def fail_write(self, *_args, **_kwargs):
        raise OSError("simulated parquet write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(TrainingRefused, match="candidate_table_write_failed"):
        refresh.append_candidates(path, pd.DataFrame([_row("new", 2)]))

    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert not list(tmp_path.glob(".candidates.parquet.*.tmp"))


def test_candidate_table_reads_and_build_schema_refusals_are_typed(tmp_path):
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not parquet")
    with pytest.raises(TrainingRefused, match="candidate_table_read_failed"):
        refresh.append_candidates(corrupt, pd.DataFrame())

    invalid = tmp_path / "invalid.parquet"
    pd.DataFrame({"wrong": [1]}).to_parquet(invalid, index=False)
    with pytest.raises(TrainingRefused, match="candidate_table_invalid"):
        refresh.append_candidates(invalid, pd.DataFrame())

    stored = tmp_path / "stored.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(stored, index=False)
    malformed = pd.DataFrame([_row("new", 2)]).drop(columns="entry_date")
    with pytest.raises(TrainingRefused, match="candidate_build_invalid"):
        refresh.append_candidates(stored, malformed)


def test_successful_append_uses_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)
    calls = []
    original_replace = refresh.os.replace

    def record_replace(source, destination):
        calls.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(refresh.os, "replace", record_replace)
    assert refresh.append_candidates(path, pd.DataFrame([_row("new", 2)])) == 1
    assert len(calls) == 1
    assert calls[0][1] == path
    assert "new" in set(pd.read_parquet(path)["event_id"])


def test_missing_candidate_table_is_typed_refusal(tmp_path):
    with pytest.raises(TrainingRefused, match="candidate_table_missing"):
        refresh.append_candidates(tmp_path / "missing.parquet", pd.DataFrame())


def test_append_accepts_entry_date_equality_and_rejects_missing_dates(tmp_path):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)
    assert refresh.append_candidates(path, pd.DataFrame([_row("equal", 2)])) == 1
    before = pd.read_parquet(path)

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(path, pd.DataFrame([_row("missing", 3, event_date="NaT")]))

    pd.testing.assert_frame_equal(pd.read_parquet(path), before)


def test_append_refuses_candidate_table_over_size_limit(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)
    monkeypatch.setattr(refresh, "CHOOSER_POOL_MAX_BYTES", 1)

    with pytest.raises(TrainingRefused, match="candidate_table_too_large"):
        refresh.append_candidates(path, pd.DataFrame())


def test_refresh_wraps_builder_failure_as_typed_refusal(tmp_path, monkeypatch):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)

    def fail_build(**_kwargs):
        raise RuntimeError("chain load failed")

    builder = SimpleNamespace(build_all=fail_build)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None))
    monkeypatch.setattr(refresh.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(refresh.importlib.util, "module_from_spec", lambda _spec: builder)

    with pytest.raises(TrainingRefused, match="candidate_build_failed"):
        refresh.refresh("2026-10", path)


def test_emit_scores_candidate_values_against_midpoint_oracle():
    import importlib.util
    import json
    import sys

    import numpy as np

    root = refresh.ROOT
    path = root / "experiments/EXP-137_one_book_per_family_which_enumerated_str/build.py"
    spec = importlib.util.spec_from_file_location("chooser_oracle_exp137", path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    ev = {
        "strikes": np.array([100.0, 95.0]),
        "bid_e": np.array([4.0, 2.0]), "ask_e": np.array([6.0, 4.0]),
        "bid_x": np.array([1.0, 3.0]), "ask_x": np.array([3.0, 5.0]),
        "expiry": pd.Timestamp("2026-10-16"), "dte_entry": 30, "dte_exit": 20,
        "spot_entry": 100.0, "spot_exit": 102.0, "pool_n": 12,
        "tradeable": np.array([True, True]),
    }
    row = {
        "event_id": "oracle", "ticker": "XYZ", "event_date": pd.Timestamp("2026-09-01"),
        "session": "AMC", "entry_date": pd.Timestamp("2026-09-01"),
        "exit_date": pd.Timestamp("2026-10-15"),
        "pred_abs_move": 5.0, "pred_abs_move_sd": 1.0,
    }
    got = builder.e133._emit(
        ev, row, "arm", np.array([0, 1, 0, 0, 0, 0, 0]),
        np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        0, 0.1, 0.2, "pattern",
    )
    mid = next(candidate for candidate in got if candidate["fill_alpha"] == 0.5)
    legs = json.loads(mid["legs"])
    assert mid["entry_cost"] == pytest.approx(5.0 - 3.0)
    assert mid["exit_value"] == pytest.approx(2.0 - 4.0)
    assert mid["pnl"] == pytest.approx(-4.0)
    assert mid["strike"] == pytest.approx(100.0)
    assert [(leg["side"], leg["strike"], leg["price"]) for leg in legs["entry"]] == [
        ("buy", 100.0, 5.0), ("sell", 95.0, 3.0),
    ]
    assert [(leg["side"], leg["strike"], leg["price"]) for leg in legs["exit"]] == [
        ("sell", 100.0, 2.0), ("buy", 95.0, 4.0),
    ]


def _load_exp137_builder():
    import importlib.util
    import sys

    path = refresh.ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/build.py"
    spec = importlib.util.spec_from_file_location("chooser_helper_exp137", path)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    return builder


def test_month_window_is_half_open_and_effective_exit_uses_expiry():
    builder = _load_exp137_builder()
    planned = pd.DataFrame({
        "event_id": ["prior", "first", "last", "next"],
        "exit_date": pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-31", "2026-11-01"]),
    })
    assert builder._month_window(planned, "2026-10")["event_id"].tolist() == ["first", "last"]

    trades = pd.DataFrame({
        "exit_arb_ok": [False, True], "exit_value": [-1.0, 2.0],
        "exit_date": pd.to_datetime(["2026-10-30", "2026-10-30"]),
        "expiry": pd.to_datetime(["2026-11-06", "2026-11-13"]),
    })
    settled = builder._effective_exit_trades(trades)
    assert settled["exit_date"].tolist() == list(pd.to_datetime(["2026-11-06", "2026-10-30"]))
    assert trades["exit_date"].tolist() == list(pd.to_datetime(["2026-10-30", "2026-10-30"]))


def test_monthly_skip_refuses_failed_inputs_but_counts_benign_structure_skips():
    builder = _load_exp137_builder()
    for code in ("structure_unresolved", "exit_structure_unresolved"):
        assert builder._refuse_monthly_event_skip(builder.e133.EventSkip(code), "event") is None
    with pytest.raises(RuntimeError, match="event: bad_quote"):
        builder._refuse_monthly_event_skip(builder.e133.EventSkip("bad_quote"), "event")

