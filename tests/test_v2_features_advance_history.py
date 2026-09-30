"""Native one-step history arithmetic, independent expectations and parity."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pandas as pd
import pytest

from engine.features import advance_history as legacy_advance_history
from engine.v2.features.panel_math import advance_history


# Trimmed, unchanged columns from the persisted research panel:
# earnings_predictions/data/processed/events_with_orats_sum.csv
# Source SHA256 c33e9b0581321670e098981acd593ceebb6fab351b755420310ee0a474a42b5f.
# Captured 2026-09-30: A/2009-05-14 and A/2011-05-13 (date is event identity).
# This older panel lacks optional mean_prior_or_implied. Its differently defined
# mean_prior_implied_move is deliberately not renamed or substituted.
@pytest.mark.needs_corpus
def test_captured_panel_rows_match_legacy_with_unavailable_fields():
    # Licensed captured values stay in the ignored private corpus. CI runs the
    # synthetic boundary cases below; this acceptance case must run locally.
    fixture = Path(os.environ.get(
        "V2_HISTORY_PANEL_FIXTURE",
        Path(__file__).resolve().parents[1] / "fixtures/v2_features/history_rows.csv"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "4faf54a9c9f82efd345d08a1660c18ca0fe988be48401acde9a2a2aa762eb760")
    frame = pd.read_csv(fixture, float_precision="round_trip")
    assert list(zip(frame.ticker, frame.date)) == [("A", "2009-05-14"), ("A", "2011-05-13")]
    for _, row in frame.iterrows():
        before = row.copy()
        actual = advance_history(row)
        _assert_same(actual, legacy_advance_history(row))
        assert actual["n_prior"] == row["n_prior"] + 1
        assert math.isnan(actual["mean_prior_or_implied"])
        assert math.isnan(actual["ema12_prior_move"]) == (row["n_prior"] == 4)
        pd.testing.assert_series_equal(row, before)


def _algebra_row():
    """Synthetic scalar boundary input; not presented as captured market data."""
    row = {
        "n_prior": 11, "move": -6.0, "abs_move": 6.0, "or_implied": 5.0,
        "mean_prior_move": 1.0, "mean_prior_abs_move": 4.0,
        "mean_prior_or_implied": 5.5,
    }
    for span in (2, 4, 8, 12):
        row[f"ema{span}_prior_move"] = 2.0
        row[f"ema{span}_prior_abs_move"] = 3.0
    return row


def _assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if math.isnan(value):
            assert math.isnan(actual[key]), key
        else:
            assert actual[key] == value, key


def test_independent_arithmetic_keys_and_input_immutability():
    row = _algebra_row()
    before = row.copy()
    expected = {
        "n_prior": 12, "mean_prior_move": 5 / 12,
        "mean_prior_abs_move": 50 / 12, "mean_prior_or_implied": 65.5 / 12,
    }
    for span in (2, 4, 8, 12):
        alpha = 2 / (span + 1)
        expected[f"ema{span}_prior_move"] = alpha * -6 + (1 - alpha) * 2
        expected[f"ema{span}_prior_abs_move"] = alpha * 6 + (1 - alpha) * 3
    result = advance_history(row)
    _assert_same(result, expected)
    assert row == before
    result["n_prior"] = -1
    _assert_same(advance_history(row), expected)


@pytest.mark.parametrize("field", ["mean_prior_move", "ema12_prior_move"])
def test_comparator_rejects_corrupted_numeric_and_missing_aggregates(field):
    row = _algebra_row()
    row["ema12_prior_move"] = float("nan")
    expected = {"mean_prior_move": 5 / 12, "ema12_prior_move": float("nan")}
    result = advance_history(row)
    actual = {key: result[key] for key in expected}
    _assert_same(actual, expected)
    actual[field] = 0.0
    with pytest.raises(AssertionError, match=field):
        _assert_same(actual, expected)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_incomplete_ema_and_means_stay_missing_at_span_boundary(missing):
    row = _algebra_row()
    row.update(ema12_prior_move=missing, ema12_prior_abs_move=missing,
               mean_prior_move=missing, mean_prior_or_implied=missing)
    result = advance_history(row)
    assert result["n_prior"] == 12
    for key in ("ema12_prior_move", "ema12_prior_abs_move", "mean_prior_move",
                "mean_prior_or_implied"):
        assert math.isnan(result[key])
    assert math.isfinite(result["ema8_prior_move"])
    _assert_same(result, legacy_advance_history(row))


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "absent"])
def test_missing_implied_observation_carries_known_mean(missing):
    row = _algebra_row()
    if isinstance(missing, str):
        row.pop("or_implied")
    else:
        row["or_implied"] = missing
    result = advance_history(row)
    assert result["mean_prior_or_implied"] == 5.5
    _assert_same(result, legacy_advance_history(row))
    row.pop("mean_prior_or_implied")
    assert math.isnan(advance_history(row)["mean_prior_or_implied"])


@pytest.mark.parametrize("field", ["n_prior", "move", "abs_move", "mean_prior_move",
                                   "mean_prior_abs_move", "ema4_prior_move"])
def test_missing_required_key_propagates(field):
    row = _algebra_row()
    row.pop(field)
    for implementation in (advance_history, legacy_advance_history):
        with pytest.raises(KeyError, match=field):
            implementation(row)


def test_scalar_errors_propagate_and_cutoff_metadata_is_not_interpreted():
    row = _algebra_row()
    row.update(date="not-a-date", as_of="caller-owned", ticker="AAPL")
    _assert_same(advance_history(row), legacy_advance_history(row))
    row["move"] = "not-a-number"
    for implementation in (advance_history, legacy_advance_history):
        with pytest.raises(ValueError):
            implementation(row)
