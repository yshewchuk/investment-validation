"""Tests for experiments/size_pnl_contrast.py (EXP-188 slice S8a).

Synthetic arrays and fixed seeds only; resamples=500 for speed.
"""

import numpy as np
import pytest

from experiments.size_pnl_contrast import (
    CLEAR_SHARE,
    bootstrap_interval,
    classify,
    contrast,
    group_signs,
    pair_status,
)

RS = 500
ARMS = ("as_is", "retuned")
QUARTERS = (
    ["2023Q3", "2023Q4"]
    + [f"2024Q{i}" for i in range(1, 5)]
    + [f"2025Q{i}" for i in range(1, 5)]
)
TICKERS = [f"T{i:02d}" for i in range(40)]


def _events(effect, seed):
    rng = np.random.default_rng(seed)
    q, t = [], []
    for quarter in QUARTERS:
        for ticker in TICKERS:
            for _ in range(4):
                q.append(quarter)
                t.append(ticker)
    d = effect + rng.normal(0.0, 1.0, size=len(q))
    return d, np.array(q), np.array(t)


def _pairs(effect, base_seed, n_pairs=3):
    pairs = {}
    for i in range(n_pairs):
        pairs[f"pair{i}"] = {}
        for j, arm in enumerate(ARMS):
            d, q, t = _events(effect, base_seed + 10 * i + j)
            pairs[f"pair{i}"][arm] = contrast(d, q, t, resamples=RS)
    return pairs


def _summary(mean, interval):
    return {
        "n_events": 1600,
        "mean": mean,
        "quarter_signs": {q: 1 for q in QUARTERS},
        "year_signs": {2023: 1, 2024: 1, 2025: 1},
        "quarter_interval": interval,
        "ticker_interval": interval,
        "quarter_counts": {"positive": len(QUARTERS), "negative": 0, "zero": 0},
    }


def test_known_effect_summary():
    d, q, t = _events(1.0, 1)
    s = contrast(d, q, t, resamples=RS)
    assert s["mean"] > 0
    assert s["quarter_interval"][0] > 0
    assert s["ticker_interval"][0] > 0
    assert set(s["quarter_signs"].values()) == {1}
    assert s["year_signs"] == {2023: 1, 2024: 1, 2025: 1}
    assert s["quarter_counts"] == {"positive": 10, "negative": 0, "zero": 0}


def test_known_effect_classify_clear_and_reversed():
    pairs = _pairs(1.0, 100)
    assert classify(pairs, 1) == "clear"
    assert classify(pairs, -1) == "reversed"


def test_null_effect_is_no_difference():
    pairs = _pairs(0.0, 200)
    assert classify(pairs, 1) == "no_difference"
    d, q, _ = _events(0.0, 5)
    _, lo, hi = bootstrap_interval(d, q, resamples=RS)
    assert lo <= 0.0 <= hi


def test_directional_from_literal_summaries():
    pairs = {
        f"pair{i}": {
            "as_is": _summary(1.0, (0.8, 1.2)),
            "retuned": _summary(0.05, (-0.1, 0.2)),
        }
        for i in range(3)
    }
    assert classify(pairs, 1) == "directional"
    assert classify(pairs, -1) == "no_difference"


def test_one_none_arm_drops_pair_and_still_classifies():
    pairs = _pairs(1.0, 300)
    pairs["pair2"]["retuned"] = None
    assert classify(pairs, 1) == "clear"


def test_two_none_arms_make_not_testable():
    pairs = _pairs(1.0, 300)
    pairs["pair1"]["as_is"] = None
    pairs["pair2"]["retuned"] = None
    assert classify(pairs, 1) == "not_testable"


def test_group_signs_tiny_hand_array():
    d = np.array([1.0, 2.0, -0.5, -4.0, 0.0])
    labels = ["2024Q1", "2024Q2", "2024Q1", "2024Q2", "2024Q3"]
    signs = group_signs(d, labels)
    assert signs == {"2024Q1": 1, "2024Q2": -1, "2024Q3": 0}
    assert list(signs) == sorted(signs)


def test_bootstrap_deterministic_for_seed_and_differs_across_seeds():
    d, q, _ = _events(0.5, 7)
    a = bootstrap_interval(d, q, resamples=RS, seed=1)
    b = bootstrap_interval(d, q, resamples=RS, seed=1)
    c = bootstrap_interval(d, q, resamples=RS, seed=2)
    assert a == b
    assert a[1:] != c[1:]


def test_ticker_and_quarter_intervals_bracket_mean():
    d, q, t = _events(0.5, 9)
    s = contrast(d, q, t, resamples=RS)
    for key in ("quarter_interval", "ticker_interval"):
        lo, hi = s[key]
        assert lo <= s["mean"] <= hi


def test_validation_raises_value_error():
    with pytest.raises(ValueError):
        group_signs([1.0, 2.0], ["2024Q1"])
    with pytest.raises(ValueError):
        group_signs([], [])
    with pytest.raises(ValueError):
        group_signs([1.0, float("nan")], ["2024Q1", "2024Q2"])


def test_missing_year_makes_years_ok_false():
    d, q, t = _events(1.0, 11)
    s = contrast(d, q, t, resamples=RS)
    assert pair_status(s, 1, share=CLEAR_SHARE)["years_ok"] is True
    broken = dict(s, year_signs=dict(s["year_signs"]))
    del broken["year_signs"][2025]
    assert pair_status(broken, 1, share=CLEAR_SHARE)["years_ok"] is False
