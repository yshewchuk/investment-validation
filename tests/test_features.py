"""Feature builders.

The load-bearing property here is that the live path and the panel path produce
the same numbers, because the models are trained on the panel and served on the
live path. :func:`advance_history` is where that property is won or lost, so it
is checked against hand-computed recursions rather than against the panel it is
meant to reproduce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import features
from engine.data.features import panel as panel_mod


class TestAdvanceHistory:
    """The one-step recursion that lets the live path resume the panel's state."""

    def row(self, **kwargs):
        base = {
            "n_prior": 10,
            "move": 6.0,
            "abs_move": 6.0,
            "implied_move": 5.0,
            "mean_prior_move": 1.0,
            "mean_prior_abs_move": 4.0,
            "mean_prior_implied_move": 5.5,
        }
        for span in panel_mod.SPANS:
            base[f"ema{span}_prior_move"] = 2.0
            base[f"ema{span}_prior_abs_move"] = 3.0
        base.update(kwargs)
        return pd.Series(base)

    def test_counter_advances(self):
        assert features.advance_history(self.row())["n_prior"] == 11

    def test_mean_is_exact(self):
        """(1.0 × 10 + 6.0) / 11."""
        out = features.advance_history(self.row())
        assert out["mean_prior_move"] == pytest.approx((1.0 * 10 + 6.0) / 11)
        assert out["mean_prior_abs_move"] == pytest.approx((4.0 * 10 + 6.0) / 11)

    def test_implied_mean_is_exact(self):
        out = features.advance_history(self.row())
        assert out["mean_prior_implied_move"] == pytest.approx((5.5 * 10 + 5.0) / 11)

    def test_ema_resumes_the_panel_recursion(self):
        out = features.advance_history(self.row())
        for span in panel_mod.SPANS:
            a = 2.0 / (span + 1.0)
            assert out[f"ema{span}_prior_move"] == pytest.approx(a * 6.0 + (1 - a) * 2.0)
            assert out[f"ema{span}_prior_abs_move"] == pytest.approx(a * 6.0 + (1 - a) * 3.0)

    def test_an_unavailable_ema_stays_unavailable(self):
        """You cannot resume a recursion you have no value to resume from."""
        out = features.advance_history(self.row(ema12_prior_move=np.nan))
        assert np.isnan(out["ema12_prior_move"])
        assert not np.isnan(out["ema2_prior_move"])

    def test_a_missing_implied_carries_the_mean_forward(self):
        out = features.advance_history(self.row(implied_move=np.nan))
        assert out["mean_prior_implied_move"] == pytest.approx(5.5)

    def test_matches_a_full_recomputation_over_a_known_history(self):
        """Advancing must agree with recomputing from scratch, given full history."""
        rng = np.random.default_rng(0)
        moves = rng.normal(0, 5, 30).tolist()
        abs_moves = [abs(m) for m in moves]
        implied = rng.uniform(3, 9, 30).tolist()

        # The panel's row for event 20, then advanced to event 21.
        at_20 = panel_mod.history_features(moves[:20], abs_moves[:20], implied[:20])
        row = pd.Series({**at_20, "move": moves[20], "abs_move": abs_moves[20],
                         "implied_move": implied[20]})
        advanced = features.advance_history(row)

        direct = panel_mod.history_features(moves[:21], abs_moves[:21], implied[:21])
        for key, value in direct.items():
            if value is None:
                continue
            assert advanced[key] == pytest.approx(value, rel=1e-12), key


class TestLiveUnavailable:
    def test_flags_the_legacy_implied_move_feature(self):
        with pytest.raises(features.UnservableFeature, match="or_implied"):
            features.assert_live_available(["ema12r_abs", "implied_move"])

    def test_passes_a_servable_list(self):
        features.assert_live_available(["ema12r_abs", "or_implied", "mcap_log"])

    def test_implied_move_is_excluded_from_nothing_but_models(self):
        """It stays a panel column — it is only barred from a champion's inputs."""
        assert "implied_move" in features.PANEL_FEATURE_COLUMNS
        assert "implied_move" in features.LIVE_UNAVAILABLE

    def test_outcomes_are_never_features(self):
        for column in features.OUTCOME_COLUMNS:
            assert column not in features.PANEL_FEATURE_COLUMNS


class TestDailyStateFrame:
    @pytest.fixture
    def daily(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        return pd.DataFrame(
            {
                "ticker": "TEST",
                "date": dates,
                "src_iv": "orats",
                "implied_move": np.arange(30, dtype=float),
                "iv10": np.arange(30, dtype=float) * 2,
                "iv30": np.arange(30, dtype=float) * 3,
                "exern_iv10": 1.0,
                "exern_iv30": np.arange(30, dtype=float),
                "iee": 1.0,
                "skew": 1.0,
                "contango": 1.0,
                "fwd90_30": 1.0,
                "fexern90_30": 1.0,
                "rvol30": 1.0,
                "spot": 100.0,
                "mcap_log": 25.0,
            }
        )

    def test_reads_the_row_on_or_before_as_of(self, daily):
        """as_of is a close we would trade at, so that close's quotes are ours."""
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 10.0

    def test_falls_back_to_the_previous_row_on_a_non_trading_date(self, daily):
        friday = daily["date"].iloc[4]
        assert friday.weekday() == 4
        saturday = friday + pd.Timedelta(days=1)
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [saturday]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 4.0

    def test_lags_count_rows_not_calendar_days(self, daily):
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im_d1"].iloc[0] == pytest.approx(1.0)
        assert out["im_d5"].iloc[0] == pytest.approx(5.0)
        assert out["iv30_d10"].iloc[0] == pytest.approx(30.0)

    def test_no_history_yields_nan_not_a_borrowed_value(self, daily):
        request = pd.DataFrame(
            {"ticker": ["TEST"], "as_of": [daily["date"].iloc[0] - pd.Timedelta(days=5)]}
        )
        out = features.daily_state_frame(request, daily=daily)
        assert np.isnan(out["im"].iloc[0])

    def test_a_lag_reaching_before_the_series_is_nan(self, daily):
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[2]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 2.0
        assert np.isnan(out["im_d5"].iloc[0])

    def test_unknown_ticker_yields_nan(self, daily):
        request = pd.DataFrame({"ticker": ["OTHER"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert np.isnan(out["im"].iloc[0])

    def test_rows_without_an_iv_source_are_not_the_answer(self, daily):
        """`daily_market` also carries cap-only rows, which have no surface."""
        extra = daily.iloc[[11]].copy()
        extra["src_iv"] = None
        for column in ("implied_move", "iv10", "iv30"):
            extra[column] = np.nan
        seeded = pd.concat([daily.iloc[:11], extra], ignore_index=True)
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[11]]})
        out = features.daily_state_frame(request, daily=seeded)
        assert out["im"].iloc[0] == 10.0

    def test_preserves_request_order(self, daily):
        request = pd.DataFrame(
            {
                "ticker": ["TEST", "TEST"],
                "as_of": [daily["date"].iloc[20], daily["date"].iloc[5]],
            }
        )
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].tolist() == [20.0, 5.0]


class TestContextColumns:
    def test_the_context_loads_every_column_its_consumers_read(self):
        """A shared loader must satisfy every consumer, not just the first one.

        `add_orats_features` reads ORATS_FEATURES and `daily_state_frame` reads
        DAILY_STATE_FIELDS; the two overlap but neither contains the other.
        Loading only one set produces a KeyError deep inside a scoring call, on
        a column nobody was thinking about.
        """
        import inspect

        source = inspect.getsource(features.FeatureContext.load)
        assert "DAILY_STATE_FIELDS" in source
        assert "ORATS_FEATURES" in source

    def test_the_two_field_sets_genuinely_differ(self):
        """If they ever coincide, the test above is guarding nothing."""
        only_daily = set(features.DAILY_STATE_FIELDS) - set(panel_mod.ORATS_FEATURES)
        assert only_daily, "DAILY_STATE_FIELDS is now a subset — revisit the loader test"
        assert {"iv10", "exern_iv10", "spot"} <= only_daily


class TestFeatureLists:
    def test_history_features_are_all_panel_columns(self):
        for name in features.EVENT_HISTORY_FEATURES:
            assert name in panel_mod.PANEL_COLUMNS, name

    def test_market_state_is_excluded_from_the_history_block(self):
        """The distinction that keeps STR-RUNUP's early entry leak-free."""
        for leaky in ("or_iv30", "dist_high", "spy_vol20", "or_implied"):
            assert leaky not in features.EVENT_HISTORY_FEATURES

    def test_daily_state_columns_cover_levels_and_lags(self):
        assert "im" in features.DAILY_STATE_COLUMNS
        assert "im_d10" in features.DAILY_STATE_COLUMNS
        # Size and price levels are not differenced.
        assert "mcap_log_d1" not in features.DAILY_STATE_COLUMNS
        assert "spot_d1" not in features.DAILY_STATE_COLUMNS
