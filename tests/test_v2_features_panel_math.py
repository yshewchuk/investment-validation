from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from engine.v2.features import panel_math
from engine.data.features.panel import (
    _anchor_index as legacy_anchor_index,
    _causal_ema as legacy_causal_ema,
    add_implied_history as legacy_add_implied_history,
    history_features as legacy_history_features,
)
from engine.features import daily_state_frame as legacy_daily_state_frame


def test_causal_ema_matches_legacy():
    for span in (2, 4, 8, 12):
        histories = [
            [],
            [1.0],
            [float(i) for i in range(1, span + 1)],
            [float(i) for i in range(1, 20)],
        ]
        for history in histories:
            assert (
                panel_math._causal_ema(history, span)
                == legacy_causal_ema(history, span)
            )


def test_history_features_matches_legacy():
    fixtures = [
        ([], []),
        ([1.0], [1.0]),
        ([1.0, -2.0], [1.0, 2.0]),
        (
            [float(i) - 10 for i in range(20)],
            [abs(float(i) - 10) for i in range(20)],
        ),
    ]
    for prior_moves, prior_abs in fixtures:
        assert panel_math.history_features(
            prior_moves, prior_abs
        ) == legacy_history_features(prior_moves, prior_abs)


def test_anchor_index_matches_legacy():
    series_dates = np.array(
        pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        ).values
    )

    cases = [
        (["2024-01-04"], None),
        (["2024-01-05"], ["2024-01-03"]),
        (["2024-01-03"], ["2024-01-08"]),
        (["2024-01-05"], ["2024-01-04"]),
    ]
    for event_strings, as_of_strings in cases:
        event_dates = np.array(pd.to_datetime(event_strings).values)
        if as_of_strings is None:
            as_of_dates = None
        else:
            as_of_dates = np.array(pd.to_datetime(as_of_strings).values)
        assert np.array_equal(
            panel_math._anchor_index(series_dates, event_dates, as_of_dates),
            legacy_anchor_index(series_dates, event_dates, as_of_dates),
        )

    before_first_event = np.array(pd.to_datetime(["2024-01-01"]).values)
    result = panel_math._anchor_index(series_dates, before_first_event, None)
    assert result[0] == -1
    assert np.array_equal(
        result,
        legacy_anchor_index(series_dates, before_first_event, None),
    )


def test_anchor_index_negative_one_is_guarded_in_legacy_callers():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    panel_path = repo_root / "engine" / "data" / "features" / "panel.py"
    with open(panel_path) as handle:
        source = handle.read()
    guards = ("if j < 0 or j >= len(closes):", "if idx < 0:", "if j < 0:")
    for guard in guards:
        assert guard in source, (
            f"legacy guard {guard!r} disappeared from {panel_path}; a future "
            "v2 caller of panel_math._anchor_index must carry the same guard "
            "before indexing with its result"
        )


def test_add_implied_history_matches_legacy_and_shifts_by_one():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-05",
                    "2024-01-10",
                    "2024-01-03",
                    "2024-01-09",
                ]
            ),
            "or_implied": [0.05, 0.07, 0.09, 0.10, 0.20],
        }
    )

    ported = panel_math.add_implied_history(df)
    legacy = legacy_add_implied_history(df)

    pd.testing.assert_frame_equal(
        ported[["mean_prior_or_implied"]],
        legacy[["mean_prior_or_implied"]],
    )

    aaa = ported[ported["ticker"] == "AAA"].reset_index(drop=True)
    bbb = ported[ported["ticker"] == "BBB"].reset_index(drop=True)
    assert math.isnan(aaa.loc[0, "mean_prior_or_implied"])
    assert math.isnan(bbb.loc[0, "mean_prior_or_implied"])
    assert aaa.loc[2, "mean_prior_or_implied"] == pytest.approx((0.05 + 0.07) / 2)


def _daily_frame_with_missing_surface_row() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",
            "2024-01-19",
            "2024-01-22",
        ]
    )
    n = len(dates)
    daily = pd.DataFrame(
        {
            "ticker": ["AAA"] * n,
            "date": dates,
            "src_iv": [1.0] * n,
            "implied_move": [0.10 + 0.01 * i for i in range(n)],
            "iv10": [0.20 + 0.01 * i for i in range(n)],
            "iv30": [0.30 + 0.01 * i for i in range(n)],
            "exern_iv10": [0.40 + 0.01 * i for i in range(n)],
            "exern_iv30": [0.50 + 0.01 * i for i in range(n)],
            "iee": [0.60 + 0.01 * i for i in range(n)],
            "skew": [0.70 + 0.01 * i for i in range(n)],
            "contango": [0.80 + 0.01 * i for i in range(n)],
            "fwd90_30": [0.90 + 0.01 * i for i in range(n)],
            "fexern90_30": [1.00 + 0.01 * i for i in range(n)],
            "rvol30": [1.10 + 0.01 * i for i in range(n)],
            "spot": [100.0 + i for i in range(n)],
            "mcap_log": [5.0 + 0.1 * i for i in range(n)],
        }
    )
    daily.loc[1, "src_iv"] = np.nan
    return daily


def _normalize_legacy_state_row(row) -> dict[str, float]:
    normalized: dict[str, float] = {}
    keys = list(panel_math.DAILY_STATE_FIELDS.values())
    for source_field in panel_math.LAGGED_FIELDS:
        output_key = panel_math.DAILY_STATE_FIELDS[source_field]
        for lag in panel_math.DAILY_STATE_LAGS:
            keys.append(f"{output_key}_d{lag}")
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        normalized[key] = float(value)
    return normalized


def test_daily_state_lookup_matches_legacy_daily_state_frame():
    daily = _daily_frame_with_missing_surface_row()
    decision_date = daily.loc[12, "date"]

    requests = pd.DataFrame({"ticker": ["AAA"], "as_of": [decision_date]})
    legacy_row = legacy_daily_state_frame(
        requests, daily=daily, as_of_column="as_of"
    ).iloc[0]

    rows = daily[daily["ticker"] == "AAA"].to_dict("records")
    ported = panel_math.daily_state_lookup(rows, decision_date)

    assert _normalize_legacy_state_row(legacy_row) == ported

    nan_surface_date = daily.loc[1, "date"]
    first_surface_date = daily.loc[0, "date"]
    assert panel_math.daily_state_lookup(
        rows, nan_surface_date
    ) == panel_math.daily_state_lookup(rows, first_surface_date)


def test_daily_state_lookup_missing_values_are_absent_keys():
    rows = [
        {
            "date": pd.Timestamp("2024-01-02"),
            "src_iv": 1.0,
            "implied_move": 0.10,
            "iv10": 0.20,
            "iv30": 0.30,
            "exern_iv10": 0.40,
            "exern_iv30": 0.50,
            "iee": 0.60,
            "skew": 0.70,
            "contango": 0.80,
            "fwd90_30": 0.90,
            "fexern90_30": 1.00,
            "rvol30": 1.10,
            "spot": 100.0,
            "mcap_log": 5.0,
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "src_iv": 1.0,
            "implied_move": 0.11,
            "iv10": 0.21,
            "iv30": 0.31,
            "exern_iv10": 0.41,
            "exern_iv30": 0.51,
            "iee": 0.61,
            "skew": 0.71,
            "contango": 0.81,
            "fwd90_30": 0.91,
            "fexern90_30": 1.01,
            "rvol30": 1.11,
            "spot": 101.0,
            "mcap_log": 5.1,
        },
    ]

    assert panel_math.daily_state_lookup(rows, pd.Timestamp("2024-01-01")) == {}

    result = panel_math.daily_state_lookup(rows, pd.Timestamp("2024-01-03"))
    assert "im_d1" in result
    for key in (
        "im_d5",
        "im_d10",
        "iv10_d5",
        "iv10_d10",
        "iv30_d5",
        "iv30_d10",
        "exern_iv30_d5",
        "exern_iv30_d10",
    ):
        assert key not in result


def test_causal_ema_and_history_features_with_nan_match_legacy():
    for span in (2, 4, 8, 12):
        history = [float(i) for i in range(1, span + 1)]
        history[1] = float("nan")
        ported = panel_math._causal_ema(list(history), span)
        legacy = legacy_causal_ema(list(history), span)
        assert (ported is None and legacy is None) or (
            math.isnan(ported) and math.isnan(legacy)
        )

    prior_moves = [1.0, float("nan"), 3.0, -2.0]
    prior_abs = [1.0, float("nan"), 3.0, 2.0]
    ported_out = panel_math.history_features(prior_moves, prior_abs)
    legacy_out = legacy_history_features(prior_moves, prior_abs)
    assert ported_out.keys() == legacy_out.keys()
    for key in ported_out:
        p, l = ported_out[key], legacy_out[key]
        if isinstance(p, float) and math.isnan(p):
            assert isinstance(l, float) and math.isnan(l)
        else:
            assert p == l


def test_anchor_index_empty_matches_legacy():
    series_dates = np.array(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).values
    )
    event_dates = np.array(pd.to_datetime([]).values)
    as_of_dates = np.array(pd.to_datetime([]).values)

    ported_no_asof = panel_math._anchor_index(series_dates, event_dates, None)
    legacy_no_asof = legacy_anchor_index(series_dates, event_dates, None)
    assert np.array_equal(ported_no_asof, legacy_no_asof)

    ported_with_asof = panel_math._anchor_index(
        series_dates, event_dates, as_of_dates
    )
    legacy_with_asof = legacy_anchor_index(series_dates, event_dates, as_of_dates)
    assert np.array_equal(ported_with_asof, legacy_with_asof)


def test_add_implied_history_empty_and_single_row_match_legacy():
    empty_df = pd.DataFrame(
        {
            "ticker": pd.Series([], dtype="object"),
            "date": pd.Series([], dtype="datetime64[ns]"),
            "or_implied": pd.Series([], dtype="float64"),
        }
    )
    ported_empty = panel_math.add_implied_history(empty_df)
    legacy_empty = legacy_add_implied_history(empty_df)
    pd.testing.assert_frame_equal(
        ported_empty[["mean_prior_or_implied"]],
        legacy_empty[["mean_prior_or_implied"]],
    )

    single_row_df = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2024-01-02"]),
            "or_implied": [0.05],
        }
    )
    ported_single = panel_math.add_implied_history(single_row_df)
    legacy_single = legacy_add_implied_history(single_row_df)
    pd.testing.assert_frame_equal(
        ported_single[["mean_prior_or_implied"]],
        legacy_single[["mean_prior_or_implied"]],
    )
    assert math.isnan(ported_single.loc[0, "mean_prior_or_implied"])


def test_daily_state_lookup_empty_and_single_row_match_legacy():
    assert panel_math.daily_state_lookup([], pd.Timestamp("2024-01-02")) == {}

    daily = _daily_frame_with_missing_surface_row()
    single_row_daily = daily.iloc[[0]].reset_index(drop=True)
    decision_date = single_row_daily.loc[0, "date"]

    requests = pd.DataFrame({"ticker": ["AAA"], "as_of": [decision_date]})
    legacy_row = legacy_daily_state_frame(
        requests, daily=single_row_daily, as_of_column="as_of"
    ).iloc[0]

    rows = single_row_daily.to_dict("records")
    ported = panel_math.daily_state_lookup(rows, decision_date)

    assert _normalize_legacy_state_row(legacy_row) == ported


def test_is_present_treats_numpy_float32_nan_as_absent():
    rows = [
        {
            "date": pd.Timestamp("2024-01-02"),
            "src_iv": np.float32("nan"),
            "implied_move": 0.10,
            "iv10": 0.20,
            "iv30": 0.30,
            "exern_iv10": 0.40,
            "exern_iv30": 0.50,
            "iee": 0.60,
            "skew": 0.70,
            "contango": 0.80,
            "fwd90_30": 0.90,
            "fexern90_30": 1.00,
            "rvol30": 1.10,
            "spot": 100.0,
            "mcap_log": 5.0,
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "src_iv": 1.0,
            "implied_move": np.float32("nan"),
            "iv10": 0.21,
            "iv30": 0.31,
            "exern_iv10": 0.41,
            "exern_iv30": 0.51,
            "iee": 0.61,
            "skew": 0.71,
            "contango": 0.81,
            "fwd90_30": 0.91,
            "fexern90_30": 1.01,
            "rvol30": 1.11,
            "spot": 101.0,
            "mcap_log": 5.1,
        },
    ]

    # The first row's src_iv is a numpy float32 NaN, so it must be treated
    # as ineligible (not a surface row) exactly like a built-in NaN would be.
    result_at_first_date = panel_math.daily_state_lookup(
        rows, pd.Timestamp("2024-01-02")
    )
    assert result_at_first_date == {}

    # The second row's implied_move is a numpy float32 NaN; it must not
    # appear as a fabricated NaN in the output.
    result_at_second_date = panel_math.daily_state_lookup(
        rows, pd.Timestamp("2024-01-03")
    )
    assert "im" not in result_at_second_date
    assert result_at_second_date.get("iv10") == 0.21
