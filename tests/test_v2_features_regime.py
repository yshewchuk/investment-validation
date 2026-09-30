"""Regime arithmetic parity; captured source data stays in a private corpus."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
import pytest

from engine import paths
from engine.data.features import panel as legacy_panel
from engine.v2.features.regime import add_regime_features
from engine.v2.parity import compare_dimension
from engine.v2.parity.tolerance import SCORE_RECORD_V1

FEATURES = ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
            "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252")


def _market(size=300):
    """Explicitly synthetic observations, not a captured market sample."""
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=size),
                         "close": 100 + np.arange(size) / 10 + np.sin(np.arange(size))})


def _legacy(events, market, tmp_path, monkeypatch, as_of_column="date"):
    path = tmp_path / "reader-placeholder"
    path.touch()
    monkeypatch.setattr(legacy_panel, "_gspc_series", lambda *_: market)
    return legacy_panel.add_regime_features(events, path, as_of_column)


def _frame_records(frame):
    # The record comparator represents an absent timestamp as None, not pandas NaT.
    return {str(i): {key: None if value is pd.NaT else value
                     for key, value in frame.iloc[i].to_dict().items()}
            for i in range(len(frame))}


def _compare_parity(actual, expected):
    assert actual.shape == expected.shape
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    pd.testing.assert_series_equal(actual.dtypes, expected.dtypes)
    return compare_dimension(_frame_records(expected), _frame_records(actual), "regime",
                             tolerance_policy=SCORE_RECORD_V1)


def _assert_parity(actual, expected):
    comparison = _compare_parity(actual, expected)
    if actual.empty:
        assert comparison["agree"] is False  # Empty populations are incomparable.
    else:
        assert comparison["agree"], comparison["finding_fields"]


def test_batch_direct_shuffle_and_immutability(tmp_path, monkeypatch):
    market = _market()
    events = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2020-08-01", "2019-01-01"]),
                           "decision": pd.to_datetime(["2020-09-01", "2020-07-02", "2019-01-01"]),
                           "ticker": ["A", "B", "C"]}, index=[7, 2, 9])
    before_events, before_market = events.copy(), market.copy()
    result = add_regime_features(events, market, as_of_column="decision")
    _assert_parity(result, _legacy(events, market, tmp_path, monkeypatch, "decision"))
    for index in events.index:
        _assert_parity(result.loc[[index]], add_regime_features(
            events.loc[[index]], market, as_of_column="decision"))
    shuffled = events.iloc[[2, 0, 1]]
    _assert_parity(result.loc[shuffled.index], add_regime_features(
        shuffled, market, as_of_column="decision"))
    pd.testing.assert_frame_equal(events, before_events)
    pd.testing.assert_frame_equal(market, before_market)


def test_anchor_is_strict_before_event_and_inclusive_at_decision():
    market = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-08"]),
                           "close": [100.0, 110.0, 105.0]})
    events = pd.DataFrame({"date": pd.to_datetime(["2020-01-08", "2020-01-10", "2020-01-10"]),
                           "decision": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-01-06"])})
    result = add_regime_features(events, market, as_of_column="decision")
    assert result.regime_asof.tolist() == list(pd.to_datetime(["2020-01-03", "2020-01-08", "2020-01-03"]))
    assert result[list(FEATURES)].isna().all().all()


def test_simple_return_sample_volatility_and_short_history():
    market = _market(6)
    market["close"] = [100.0, 110.0, 99.0, 118.8, 106.92, 117.612]
    events = pd.DataFrame({"date": pd.to_datetime(["2020-01-06", "2020-01-07"])})
    result = add_regime_features(events, market)
    expected = statistics.stdev([0.1, -0.1, 0.2, -0.1, 0.1]) * math.sqrt(252) * 100
    assert math.isnan(result.spy_vol5.iloc[0])
    assert result.spy_vol5.iloc[1] == pytest.approx(expected)
    assert result.spy_vol20.isna().all()


def test_return_and_drawdown_window_endpoints():
    market = _market(253)
    market["close"] = np.arange(100.0, 353.0)
    market.loc[0, "close"] = 10000.0  # Must not enter the trailing 252-close maximum.
    market.loc[252, "close"] = 200.0
    result = add_regime_features(pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]}), market).iloc[0]
    for window in (21, 63, 252):
        assert result[f"spy_ret{window}"] == (200 / market.close.iloc[252 - window] - 1) * 100
    assert result.spy_dd252 == (200 / 351 - 1) * 100
    assert result.regime_asof == market.date.iloc[-1]


@pytest.mark.parametrize("market_size,event_count", [(0, 1), (3, 0), (3, 1)])
def test_empty_inputs_and_no_eligible_observation(market_size, event_count, tmp_path, monkeypatch):
    market = _market(market_size)
    events = pd.DataFrame({"date": pd.to_datetime(["2019-01-01"] * event_count)})
    result = add_regime_features(events, market)
    _assert_parity(result, _legacy(events, market, tmp_path, monkeypatch))
    assert result[list(FEATURES)].isna().all().all()
    assert result.regime_asof.isna().all()


@pytest.mark.parametrize("value", [100.0, 0.0, float("nan")])
def test_constant_zero_and_nan_sources_preserve_legacy_arithmetic(value, tmp_path, monkeypatch):
    market = _market()
    market["close"] = value
    events = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]})
    with np.errstate(divide="ignore", invalid="ignore"):
        result = add_regime_features(events, market)
        _assert_parity(result, _legacy(events, market, tmp_path, monkeypatch))
    assert math.isnan(result.spy_vol20_rel252.iloc[0])


@pytest.mark.parametrize("field", FEATURES)
@pytest.mark.parametrize("corruption", [42.0, np.nan])
def test_comparator_rejects_numeric_missing_defects(field, corruption, tmp_path, monkeypatch):
    market = _market()
    events = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]})
    expected = _legacy(events, market, tmp_path, monkeypatch)
    actual = add_regime_features(events, market)
    _assert_parity(actual, expected)
    actual.loc[0, field] = corruption
    comparison = _compare_parity(actual, expected)
    assert comparison["agree"] is False
    assert any(path.endswith(f".{field}") for path in comparison["finding_fields"])


@pytest.mark.parametrize("corruption", [pd.Timestamp("2021-01-01"), pd.NaT])
def test_comparator_rejects_regime_asof_defect(corruption, tmp_path, monkeypatch):
    market = _market()
    events = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]})
    expected = _legacy(events, market, tmp_path, monkeypatch)
    actual = add_regime_features(events, market)
    _assert_parity(actual, expected)
    actual.loc[0, "regime_asof"] = corruption
    comparison = _compare_parity(actual, expected)
    assert comparison["agree"] is False
    assert any(path.endswith(".regime_asof") for path in comparison["finding_fields"])


def test_missing_columns_and_invalid_closes_propagate():
    events = pd.DataFrame({"date": [pd.Timestamp("2021-01-01")]})
    market = _market()
    with pytest.raises(KeyError, match="close"):
        add_regime_features(events, market.drop(columns="close"))
    with pytest.raises(KeyError, match="decision"):
        add_regime_features(events, market, as_of_column="decision")
    market["close"] = "not-a-number"
    with pytest.raises(ValueError):
        add_regime_features(events, market)


@pytest.mark.needs_corpus  # Captured benchmark observations are private, absent in CI.
def test_private_captured_regime_source_parity():
    default = paths.ROOT / "fixtures/native_regime/gspc_daily.csv"
    override = os.environ.get("V2_REGIME_CORPUS_CSV")
    path = Path(override) if override else default
    if not path.exists() and not override:
        pytest.skip("private regime corpus unavailable; set V2_REGIME_CORPUS_CSV")
    manifest = json.loads(path.with_name("manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    source_hash = manifest["source_sha256"]
    print(f"regime corpus sha256={source_hash}")
    market = pd.read_csv(path, skiprows=3, header=None,
                         names=["date", "adj", "close", "high", "low", "open", "volume"])
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values("date")
    assert len(market) >= 253
    selected = market.date.iloc[[0, 21, 63, 252]].reset_index(drop=True)
    events = pd.DataFrame({"date": selected + pd.Timedelta(days=1), "decision": selected})
    actual = add_regime_features(events, market, as_of_column="decision")
    expected = legacy_panel.add_regime_features(events, path, "decision")
    _assert_parity(actual, expected)
    pd.testing.assert_series_equal(actual.regime_asof, selected.astype("datetime64[ns]"), check_names=False)
