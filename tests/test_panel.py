"""Tier-3 panel construction — above all, leak discipline.

Every feature must be computable from information available strictly before the
print. These tests fix that property in place, because a leak is the one class
of bug that makes results look *better* and so never announces itself.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.data.features.panel import (
    MIN_HISTORY,
    ORATS_FEATURES,
    SPANS,
    _causal_ema,
    add_orats_features,
    add_regime_features,
    add_runup_features,
    build_events,
)


class TestCausalEma:
    def test_returns_none_until_the_span_is_filled(self):
        assert _causal_ema([1.0, 2.0], 4) is None
        assert _causal_ema([1.0, 2.0, 3.0, 4.0], 4) is not None

    def test_matches_the_legacy_recursion(self):
        history = [5.0, 3.0, 8.0, 2.0, 6.0]
        span = 4
        alpha = 2.0 / (span + 1.0)
        expected = history[0]
        for value in history[1:]:
            expected = alpha * value + (1 - alpha) * expected
        assert _causal_ema(history, span) == pytest.approx(expected)

    def test_is_not_the_pandas_ewm_default(self):
        # `pandas.ewm(span=n)` differs in seeding and adjustment. Substituting
        # it would silently change every stored EMA in the panel.
        history = [5.0, 3.0, 8.0, 2.0, 6.0, 1.0]
        pandas_value = pd.Series(history).ewm(span=4, adjust=True).mean().iloc[-1]
        assert _causal_ema(history, 4) != pytest.approx(pandas_value)

    def test_only_sees_the_history_it_is_given(self):
        base = [5.0, 3.0, 8.0, 2.0]
        assert _causal_ema(base, 4) == _causal_ema(list(base), 4)
        assert _causal_ema(base + [99.0], 4) != _causal_ema(base, 4)


class TestBuildEvents:
    @pytest.fixture
    def moves_dir(self, tmp_path):
        directory = tmp_path / "moves"
        directory.mkdir()
        dates = [f"20{y:02d}-01-15" for y in range(10, 20)]
        payload = {
            "ticker": "AAA",
            "data": {
                "dates": dates,
                "realized_moves": [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
                "abs_realized_moves": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "implied_moves": [2.0] * 10,
                "quarters": [1] * 10,
            },
        }
        (directory / "moves_AAA.json").write_text(json.dumps(payload))
        return directory

    def test_events_below_min_history_are_not_admitted(self, moves_dir):
        panel = build_events(moves_dir)
        assert panel["n_prior"].min() >= MIN_HISTORY
        assert len(panel) == 10 - MIN_HISTORY

    def test_history_features_exclude_the_current_event(self, moves_dir):
        panel = build_events(moves_dir).sort_values("k").reset_index(drop=True)
        first = panel.iloc[0]
        # k=4 → prior events are k=0..3 with |move| 1,2,3,4 → mean 2.5.
        assert first["k"] == 4
        assert first["mean_prior_abs_move"] == pytest.approx(2.5)
        # The current event's own |move| (5.0) must not be in it.
        assert first["abs_move"] == 5.0

    def test_every_declared_span_produces_a_column(self, moves_dir):
        panel = build_events(moves_dir)
        for span in SPANS:
            assert f"ema{span}_prior_move" in panel.columns
            assert f"ema{span}_prior_abs_move" in panel.columns

    def test_a_ragged_ticker_is_skipped_not_silently_truncated(self, tmp_path):
        directory = tmp_path / "moves"
        directory.mkdir()
        (directory / "moves_BAD.json").write_text(
            json.dumps(
                {
                    "ticker": "BAD",
                    "data": {
                        "dates": ["2020-01-01", "2020-04-01"],
                        "realized_moves": [1.0],
                        "abs_realized_moves": [1.0, 2.0],
                        "implied_moves": [2.0, 2.0],
                        "quarters": [1, 2],
                    },
                }
            )
        )
        assert build_events(directory).empty

    def test_missing_source_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_events(tmp_path / "absent")


class TestRegimeFeatures:
    @pytest.fixture
    def gspc(self, tmp_path):
        path = tmp_path / "gspc.csv"
        dates = pd.bdate_range("2018-01-01", periods=600)
        closes = 1000 * np.exp(np.cumsum(np.full(len(dates), 0.0005)))
        lines = ["Price,Adj Close,Close,High,Low,Open,Volume",
                 "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC",
                 "Date,,,,,,"]
        for d, c in zip(dates, closes):
            lines.append(f"{d.date()},{c},{c},{c},{c},{c},1000")
        path.write_text("\n".join(lines))
        return path

    def test_uses_the_last_close_strictly_before_the_event(self, gspc):
        # The event-date close must not be visible: a BMO print happens before
        # it exists, and the panel treats both sessions conservatively.
        dates = pd.bdate_range("2018-01-01", periods=600)
        event = dates[400]
        frame = pd.DataFrame({"ticker": ["AAA"], "date": [event]})
        out = add_regime_features(frame, gspc)

        raw = pd.read_csv(gspc, skiprows=3, header=None)
        closes = raw[2].to_numpy(dtype=float)
        expected = (closes[399] / closes[399 - 21] - 1.0) * 100
        assert out["spy_ret21"].iloc[0] == pytest.approx(expected)

    def test_volatility_uses_simple_returns_with_ddof_one(self, gspc):
        # Recovered from the legacy panel by search. Log returns or ddof=0 shift
        # every value ~2.6%, in one of the nine champion-model features.
        dates = pd.bdate_range("2018-01-01", periods=600)
        frame = pd.DataFrame({"ticker": ["AAA"], "date": [dates[400]]})
        out = add_regime_features(frame, gspc)

        raw = pd.read_csv(gspc, skiprows=3, header=None)
        closes = raw[2].to_numpy(dtype=float)
        simple = closes[1:] / closes[:-1] - 1.0
        expected = simple[399 - 20 : 399].std(ddof=1) * np.sqrt(252) * 100
        assert out["spy_vol20"].iloc[0] == pytest.approx(expected)

    def test_drawdown_uses_a_252_observation_window(self, gspc):
        dates = pd.bdate_range("2018-01-01", periods=600)
        frame = pd.DataFrame({"ticker": ["AAA"], "date": [dates[400]]})
        out = add_regime_features(frame, gspc)

        raw = pd.read_csv(gspc, skiprows=3, header=None)
        closes = raw[2].to_numpy(dtype=float)
        expected = (closes[399] / closes[399 - 251 : 400].max() - 1.0) * 100
        assert out["spy_dd252"].iloc[0] == pytest.approx(expected)

    def test_events_before_the_series_get_nulls_not_zeros(self, gspc):
        frame = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2017-01-01")]})
        out = add_regime_features(frame, gspc)
        assert pd.isna(out["spy_ret21"].iloc[0])


class TestRunupFeatures:
    def test_signed_streak_counts_the_run_that_ended_before_the_event(self, tmp_path):
        frame = pd.DataFrame(
            {
                "ticker": ["AAA"] * 5,
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01", "2021-01-01"]
                ),
                "move": [1.0, 2.0, 3.0, -1.0, 2.0],
                "n_prior": [4, 5, 6, 7, 8],
                "ema12_prior_abs_move": [np.nan] * 5,
                "mean_prior_abs_move": [1.0] * 5,
            }
        )
        out = add_runup_features(frame, tmp_path)
        # First event of a ticker has no prior run.
        assert out["signed_streak"].iloc[0] == 0
        # After +,+ the run entering event 3 is length 2, positive.
        assert out["signed_streak"].iloc[2] == 2
        # After the sign flips, the run entering event 5 is length 1, negative.
        assert out["signed_streak"].iloc[4] == -1

    def test_ema12r_falls_back_to_the_mean_below_twelve_priors(self, tmp_path):
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": pd.to_datetime(["2020-01-01", "2020-04-01"]),
                "move": [1.0, 2.0],
                "n_prior": [5, 15],
                "ema12_prior_abs_move": [np.nan, 7.0],
                "mean_prior_abs_move": [3.0, 4.0],
            }
        )
        out = add_runup_features(frame, tmp_path)
        assert out["ema12r_abs"].iloc[0] == 3.0  # too little history → the mean
        assert out["ema12r_abs"].iloc[1] == 7.0  # enough history → the EMA

    def test_a_ticker_with_no_price_file_gets_nulls(self, tmp_path):
        frame = pd.DataFrame(
            {
                "ticker": ["ZZZ"],
                "date": pd.to_datetime(["2020-01-01"]),
                "move": [1.0],
                "n_prior": [5],
                "ema12_prior_abs_move": [np.nan],
                "mean_prior_abs_move": [3.0],
            }
        )
        out = add_runup_features(frame, tmp_path)
        assert pd.isna(out["dist_high"].iloc[0])
        assert pd.isna(out["ret20"].iloc[0])


class TestOratsFeatures:
    def _daily(self):
        dates = pd.bdate_range("2023-01-02", periods=400)
        frame = pd.DataFrame(
            {
                "ticker": "AAA",
                "date": dates,
                "src_iv": "orats.summaries",
                "mcap_usd": np.linspace(1e10, 2e10, len(dates)),
                "mcap_log": np.log(np.linspace(1e10, 2e10, len(dates))),
            }
        )
        for column in ORATS_FEATURES:
            frame[column] = np.linspace(10.0, 50.0, len(dates))
        return frame

    def test_reads_the_last_row_strictly_before_the_event(self):
        daily = self._daily()
        event = daily["date"].iloc[100]
        panel = pd.DataFrame({"ticker": ["AAA"], "date": [event]})
        out = add_orats_features(panel, daily)
        # Index 99, not 100: the event-date row must not be visible.
        assert out["or_iv30"].iloc[0] == pytest.approx(daily["iv30"].iloc[99])

    def test_an_event_before_any_daily_row_gets_nulls(self):
        daily = self._daily()
        panel = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2020-01-01")]})
        out = add_orats_features(panel, daily)
        assert pd.isna(out["or_iv30"].iloc[0])
        assert pd.isna(out["mcap_log"].iloc[0])

    def test_market_cap_resolves_over_rows_that_have_one(self):
        # Summaries and cores do not share a date grid; tying the cap to the IV
        # observation returns a stale figure, or none at all.
        daily = self._daily()
        daily.loc[daily.index[50:], "mcap_usd"] = np.nan
        daily.loc[daily.index[50:], "mcap_log"] = np.nan
        event = daily["date"].iloc[100]
        out = add_orats_features(pd.DataFrame({"ticker": ["AAA"], "date": [event]}), daily)
        assert out["mcap_log"].iloc[0] == pytest.approx(daily["mcap_log"].iloc[49])
        assert out["mcap_asof"].iloc[0] == daily["date"].iloc[49]

    def test_cores_only_rows_never_answer_an_iv_lookup(self):
        daily = self._daily()
        # A market-cap observation on a date summaries has no row for.
        extra = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": [daily["date"].iloc[99] + pd.Timedelta(hours=12)],
                "src_iv": [None],
                "mcap_usd": [5e10],
                "mcap_log": [np.log(5e10)],
                **{c: [np.nan] for c in ORATS_FEATURES},
            }
        )
        daily = pd.concat([daily, extra], ignore_index=True).sort_values("date")
        event = daily["date"].iloc[101]
        out = add_orats_features(pd.DataFrame({"ticker": ["AAA"], "date": [event]}), daily)
        assert not pd.isna(out["or_iv30"].iloc[0])

    def test_the_z_score_standardizes_within_the_name(self):
        daily = self._daily()
        rng = np.random.default_rng(0)
        daily["exern_iv30"] = 30 + rng.normal(0, 5, len(daily))
        event = daily["date"].iloc[300]
        out = add_orats_features(pd.DataFrame({"ticker": ["AAA"], "date": [event]}), daily)

        window = daily["exern_iv30"].to_numpy()[max(0, 299 - 252) : 299]
        expected = (daily["exern_iv30"].iloc[299] - window.mean()) / window.std()
        assert out["or_exern_z252"].iloc[0] == pytest.approx(expected)

    def test_too_short_a_window_yields_no_z_score(self):
        daily = self._daily().head(30)
        event = daily["date"].iloc[29]
        out = add_orats_features(pd.DataFrame({"ticker": ["AAA"], "date": [event]}), daily)
        assert pd.isna(out["or_exern_z252"].iloc[0])
