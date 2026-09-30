"""Independent runup algebra, causal boundaries, and real legacy parity."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data.features import panel as legacy
from engine.v2.features.runup_math import add_runup_features


MARKET = ["dist_high", "dist_ema", "ret5", "ret10", "ret20"]
OUTPUTS = ["signed_streak", "ema12r_abs", *MARKET, "runup_asof"]


def _prices():
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=400),
                         "close_adj": 100.0 + np.arange(400)})


def _events(indices=(301, 311, 321, 331, 341, 351, 361)):
    prices = _prices()
    n = len(indices)
    return pd.DataFrame({
        "ticker": ["A"] * n, "date": prices.date.iloc[list(indices)].to_numpy(),
        "move": [2, 3, -1, -2, 0, 4, np.nan][:n], "n_prior": np.arange(n) + 8,
        "ema12_prior_abs_move": np.arange(n) + 30.0,
        "mean_prior_abs_move": np.arange(n) + 10.0,
    })


def _assert_equal(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_independent_algebra_and_sorted_immutable_inputs():
    events = _events()
    other = events.assign(ticker="B", move=[-1, -2, 1, 1, 1, 0, np.nan])
    frame = pd.concat([other, events]).sample(frac=1, random_state=7)
    prices = _prices().sample(frac=1, random_state=9)
    before, prices_before = frame.copy(deep=True), prices.copy(deep=True)
    actual = add_runup_features(frame, {"A": prices, "B": prices}, "date")
    assert actual.ticker.tolist() == ["A"] * 7 + ["B"] * 7
    assert actual.index.tolist() == list(range(14))
    assert actual.signed_streak.tolist() == [0, 1, 2, -1, -2, 0, 1,
                                           0, -1, -2, 1, 2, 3, 0]
    assert actual.ema12r_abs.tolist() == [10, 11, 12, 13, 34, 35, 36] * 2
    q = 1 - 2 / 253
    for offset in (0, 7):
        for j, idx in enumerate(range(300, 361, 10)):
            row = actual.iloc[offset + j]
            ema = 100 + idx - q * (1 - q ** idx) / (1 - q)
            assert row.dist_high == 0
            assert row.dist_ema == pytest.approx(((100 + idx) / ema - 1) * 100)
            for horizon in (5, 10, 20):
                assert row[f"ret{horizon}"] == pytest.approx(
                    ((100 + idx) / (100 + idx - horizon) - 1) * 100)
            assert row.runup_asof == _prices().date.iloc[idx]
    _assert_equal(frame, before)
    _assert_equal(prices, prices_before)


def test_both_date_ceilings_negative_anchor_and_post_cutoff_invariance():
    frame = _events((301, 311, 321))
    px = _prices()
    frame["decision"] = [px.date.iloc[280], px.date.iloc[330], px.date.iloc[0] - pd.Timedelta(days=1)]
    actual = add_runup_features(frame, {"A": px}, "decision")
    assert actual.runup_asof.iloc[:2].tolist() == [px.date.iloc[280], px.date.iloc[310]]
    assert pd.isna(actual.runup_asof.iloc[2])
    assert actual.loc[2, MARKET].isna().all()
    changed = px.copy()
    changed.loc[311:, "close_adj"] = -9999
    _assert_equal(add_runup_features(frame, {"A": changed}, "decision"), actual)


@pytest.mark.parametrize("price_kind", ["absent", "short", "missing_close"])
def test_unavailable_prices_keep_history_features(price_kind):
    frame = _events()
    px = _prices()
    supplied = {"A": px.iloc[:299]} if price_kind == "short" else {}
    if price_kind == "missing_close":
        supplied = {"A": px.drop(columns="close_adj")}
    actual = add_runup_features(frame, supplied, "date")
    assert actual[MARKET + ["runup_asof"]].isna().all().all()
    assert actual.signed_streak.tolist() == [0, 1, 2, -1, -2, 0, 1]
    assert actual.ema12r_abs.tolist() == [10, 11, 12, 13, 34, 35, 36]


def test_history_sufficiency_and_individual_feature_boundaries():
    frame = _events((1, 20, 21, 252, 253))
    actual = add_runup_features(frame, {"A": _prices().iloc[:300]}, "date")
    assert actual.runup_asof.notna().all()
    assert actual.loc[:1, MARKET].isna().all().all()
    assert actual.loc[2, ["ret5", "ret10", "ret20"]].notna().all()
    assert actual.loc[:3, ["dist_high", "dist_ema"]].isna().all().all()
    assert actual.loc[4, MARKET].notna().all()
    missing = add_runup_features(frame, {"A": _prices().iloc[:299]}, "date")
    assert missing[MARKET + ["runup_asof"]].isna().all().all()


@pytest.mark.parametrize("field", ["ticker", "date", "move", "n_prior",
                                   "ema12_prior_abs_move", "mean_prior_abs_move", "decision"])
def test_missing_required_inputs_raise_even_without_prices(field):
    frame = _events().assign(decision=lambda x: x.date)
    with pytest.raises(KeyError, match=field):
        add_runup_features(frame.drop(columns=field), {}, "decision")


def test_decision_argument_is_mandatory_and_invalid_values_propagate():
    with pytest.raises(TypeError, match="as_of_column"):
        add_runup_features(_events(), {})
    frame = _events()
    frame["move"] = "invalid"
    with pytest.raises(ValueError):
        add_runup_features(frame, {}, "date")
    with pytest.raises((TypeError, ValueError)):
        add_runup_features(_events().assign(decision="invalid"), {"A": _prices()}, "decision")
    with pytest.raises(ValueError):
        add_runup_features(_events(), {"A": _prices().assign(close_adj="invalid")}, "date")


def _legacy_result(frame, prices, tmp_path, monkeypatch, decision="date"):
    def no_fallback(_ticker):
        raise AssertionError("legacy must use the supplied price CSV")
    monkeypatch.setattr(legacy, "_yf_history_from_tier1", no_fallback)
    prices.to_csv(tmp_path / "px_A.csv", index=False)
    return legacy.add_runup_features(frame, px_dir=tmp_path, as_of_column=decision)


@pytest.mark.parametrize("edge", ["ordinary", "zero_return_base", "nan_close", "negative_prices"])
def test_synthetic_legacy_parity_preserves_numeric_edge_behavior(edge, tmp_path, monkeypatch):
    frame, px = _events(), _prices()
    frame["decision"] = frame.date - pd.Timedelta(days=4)
    if edge == "zero_return_base":
        px.loc[277, "close_adj"] = 0
    elif edge == "nan_close":
        px.loc[297, "close_adj"] = np.nan
    elif edge == "negative_prices":
        px["close_adj"] *= -1
    expected = _legacy_result(frame, px, tmp_path, monkeypatch, "decision")
    actual = add_runup_features(frame, {"A": px}, "decision")
    _assert_equal(actual, expected)


@pytest.mark.parametrize("field", ["ret5", "runup_asof"])
def test_comparator_rejects_wrong_field_and_future_anchor(field, tmp_path, monkeypatch):
    frame, px = _events(), _prices()
    expected = _legacy_result(frame, px, tmp_path, monkeypatch)
    actual = add_runup_features(frame, {"A": px}, "date")
    _assert_equal(actual, expected)
    actual.loc[0, field] = frame.date.iloc[0] if field == "runup_asof" else -999.0
    with pytest.raises(AssertionError, match=field):
        _assert_equal(actual, expected)


# Captured 2026-09-30 by selecting unchanged source columns and ticker A.
# events_with_orats_sum.csv source SHA256:
# c33e9b0581321670e098981acd593ceebb6fab351b755420310ee0a474a42b5f
# earnings_predictions/data/raw/yfinance/px_A.csv source SHA256:
# 188f654a8cdc285f8e4d0a36601549c36c29fbd78bb2b891f883ff0006b587a4
# Licensed values stay in ignored fixtures/v2_features; public CI uses algebra.
@pytest.mark.needs_corpus
def test_captured_real_inputs_match_legacy(tmp_path, monkeypatch):
    root = Path(os.environ.get("V2_RUNUP_FIXTURE_DIR",
                Path(__file__).resolve().parents[1] / "fixtures/v2_features"))
    hashes = {"runup_events.csv": "a379dc09ace8b8e9740f57ca64d44f85b6f31ed8e579ba62f4a88e91c8d90c62",
              "runup_prices.csv": "8040978e9cd7dcb154222b7ac0055ee65333f61e5f40ee3b81e5ef80694e57b4"}
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    frame = pd.read_csv(root / "runup_events.csv", parse_dates=["date"], float_precision="round_trip")
    px = pd.read_csv(root / "runup_prices.csv", parse_dates=["date"], float_precision="round_trip")
    assert len(frame) == 68 and len(px) == 6733
    before, price_before = frame.copy(deep=True), px.copy(deep=True)
    for decision in ("date", "decision"):
        inputs = frame if decision == "date" else frame.assign(decision=frame.date - pd.Timedelta(days=14))
        expected = _legacy_result(inputs, px, tmp_path, monkeypatch, decision)
        actual = add_runup_features(inputs, {"A": px}, decision)
        _assert_equal(actual, expected)
        assert actual[MARKET].notna().any().all()
        assert actual.runup_asof.notna().any()
    _assert_equal(frame, before)
    _assert_equal(px, price_before)
