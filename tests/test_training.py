"""The walk-forward harness and the model definitions.

The harness is where a model could accidentally grade itself on data it trained
on, so the leak-shaped properties are tested directly: a year is never in its
own training set, and a model that can see the future scores perfectly while one
that cannot does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.training import common
from engine.models.training import gate as gate_mod
from engine.models.training import implied_t1 as implied_mod
from engine.models.training import size_model as size_mod


class Mean:
    """Predicts the training mean — no features, so no accidental skill."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def fit_mean(X, y, seed):
    return Mean(float(np.mean(y)))


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(2015, 2025), 200)
    x = rng.normal(0, 1, len(years))
    return pd.DataFrame(
        {
            "year": years,
            "f1": x,
            "f2": rng.normal(0, 1, len(years)),
            "target": 2.0 * x + rng.normal(0, 0.3, len(years)),
        }
    )


class TestMetrics:
    def test_perfect_prediction(self):
        stats = common.regression_metrics([1, 2, 3], [1, 2, 3])
        assert stats["r"] == pytest.approx(1.0)
        assert stats["mae"] == 0.0

    def test_constant_prediction_has_no_correlation(self):
        """None, not a NaN that later formats as a number."""
        assert common.regression_metrics([1, 2, 3], [2, 2, 2])["r"] is None

    def test_bias_is_signed(self):
        assert common.regression_metrics([1, 2, 3], [2, 3, 4])["bias"] == pytest.approx(1.0)

    def test_ignores_non_finite_pairs(self):
        stats = common.regression_metrics([1, 2, np.nan], [1, 2, 5])
        assert stats["n"] == 2

    def test_too_few_points_reports_none(self):
        assert common.regression_metrics([1], [1])["r"] is None

    def test_decile_spread_is_positive_for_a_good_ranking(self):
        y = np.arange(200, dtype=float)
        assert common.decile_spread(y, y) > 0

    def test_decile_spread_needs_enough_rows(self):
        assert common.decile_spread([1, 2, 3], [1, 2, 3]) is None


class TestWalkForward:
    def test_a_year_is_never_in_its_own_training_set(self, panel):
        seen = {}

        def spy(X, y, seed):
            seen[len(seen)] = len(y)
            return Mean(float(np.mean(y)))

        result = common.walk_forward(
            panel, ["f1", "f2"], "target", spy, first_test_year=2016, min_train_rows=100
        )
        # Training rows grow by exactly one year's worth each step.
        sizes = [seen[i] for i in sorted(seen)]
        assert sizes == sorted(sizes)
        assert all(b - a == 200 for a, b in zip(sizes, sizes[1:]))
        assert result.years == tuple(range(2016, 2025))

    def test_a_model_that_cannot_see_the_future_has_no_skill(self, panel):
        result = common.walk_forward(
            panel, ["f1", "f2"], "target", fit_mean, first_test_year=2016, min_train_rows=100
        )
        assert abs(result.metrics["r"] or 0.0) < 0.1

    def test_a_real_model_recovers_the_signal(self, panel):
        from sklearn.linear_model import LinearRegression

        result = common.walk_forward(
            panel,
            ["f1", "f2"],
            "target",
            lambda X, y, s: LinearRegression().fit(X, y),
            first_test_year=2016,
            min_train_rows=100,
        )
        assert result.metrics["r"] > 0.95

    def test_residuals_are_signed_so_they_can_be_added_to_a_prediction(self, panel):
        result = common.walk_forward(
            panel, ["f1", "f2"], "target", fit_mean, first_test_year=2016, min_train_rows=100
        )
        expected = result.frame["target"] - result.frame["pred"]
        assert np.allclose(result.residuals, expected)

    def test_incomplete_rows_are_dropped_not_imputed(self, panel):
        panel.loc[:99, "f1"] = np.nan
        result = common.walk_forward(
            panel, ["f1", "f2"], "target", fit_mean, first_test_year=2016, min_train_rows=100
        )
        assert len(result.frame) < len(panel)
        assert result.frame["f1"].notna().all()

    def test_years_below_the_training_floor_are_skipped(self, panel):
        result = common.walk_forward(
            panel, ["f1", "f2"], "target", fit_mean, min_train_rows=100_000
        )
        assert result.years == ()
        assert result.metrics["n"] == 0

    def test_missing_columns_are_an_error(self, panel):
        with pytest.raises(KeyError, match="missing"):
            common.walk_forward(panel, ["f1", "absent"], "target", fit_mean)

    def test_by_year_table_reports_train_size(self, panel):
        result = common.walk_forward(
            panel, ["f1", "f2"], "target", fit_mean, first_test_year=2016, min_train_rows=100
        )
        assert set(result.by_year.columns) >= {"year", "n_train", "n", "r", "mae"}


class TestFitFinal:
    def test_uses_every_complete_row(self, panel):
        model = common.fit_final(panel, ["f1", "f2"], "target", fit_mean)
        assert model.value == pytest.approx(panel["target"].mean())


class TestBlendModel:
    def test_averages_its_components(self):
        blend = common.BlendModel(models=(Mean(1.0), Mean(3.0)))
        assert blend.predict(np.zeros((4, 2))).tolist() == [2.0] * 4


class TestSizeModel:
    def test_the_champion_list_is_servable(self):
        from engine.features import assert_live_available

        assert_live_available(size_mod.FEATURES)

    def test_the_legacy_list_is_not(self):
        from engine.features import UnservableFeature, assert_live_available

        with pytest.raises(UnservableFeature):
            assert_live_available(size_mod.LEGACY_FEATURES)

    def test_or_implied_replaces_implied_move(self):
        assert "or_implied" in size_mod.FEATURES
        assert "implied_move" not in size_mod.FEATURES
        assert "implied_move" in size_mod.LEGACY_FEATURES

    def test_prepare_masks_out_of_range_values(self):
        frame = pd.DataFrame(
            {"or_implied": [5.0, 999.0], "or_rvol30": [30.0, 30.0],
             "abs_move": [3.0, 3.0], "n_prior": [10, 10]}
        )
        out = size_mod.prepare(frame)
        assert out["or_implied"].tolist()[0] == 5.0
        assert np.isnan(out["or_implied"].iloc[1])

    def test_prepare_drops_thin_history(self):
        frame = pd.DataFrame({"n_prior": [1, 10], "abs_move": [1.0, 1.0]})
        assert len(size_mod.prepare(frame)) == 1

    def test_fit_returns_a_two_model_blend(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (300, len(size_mod.FEATURES)))
        y = X[:, 0] * 2 + rng.normal(0, 0.2, 300)
        model = size_mod.fit(X, y, seed=1)
        assert len(model.models) == 2
        assert model.predict(X[:5]).shape == (5,)

    def test_fit_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (300, len(size_mod.FEATURES)))
        y = X[:, 0] * 2 + rng.normal(0, 0.2, 300)
        a = size_mod.fit(X, y, seed=7).predict(X[:20])
        b = size_mod.fit(X, y, seed=7).predict(X[:20])
        assert np.allclose(a, b)


class TestImpliedT1:
    def test_features_exclude_the_panel_market_block(self):
        """Those are read at the close being predicted."""
        for leaky in ("or_iv30", "or_implied", "dist_high", "spy_vol20"):
            assert leaky not in implied_mod.FEATURES

    def test_features_are_servable(self):
        from engine.features import assert_live_available

        assert_live_available(implied_mod.FEATURES)

    def test_decision_days_include_the_runup_entry(self):
        assert 14 in implied_mod.DECISION_DAYS

    def test_days_before_print_is_a_feature_not_bookkeeping(self):
        assert "days_before_print" in implied_mod.FEATURES


class TestGate:
    def test_trains_on_mid_fills(self):
        assert gate_mod.GATE_ALPHA == 0.5

    def test_features_are_as_of_entry_only(self):
        for leaky in ("or_iv30", "or_implied", "dist_high", "spy_vol20", "abs_move"):
            assert leaky not in gate_mod.FEATURES

    def test_threshold_is_the_top_fraction_quantile(self):
        scores = np.arange(100, dtype=float)
        threshold = gate_mod.choose_threshold(scores, top_fraction=0.20)
        assert (scores >= threshold).sum() == pytest.approx(20, abs=1)

    def test_threshold_ignores_non_finite_scores(self):
        scores = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        assert np.isfinite(gate_mod.choose_threshold(scores))

    def test_empty_scores_give_nan_not_a_crash(self):
        assert np.isnan(gate_mod.choose_threshold([]))

    def test_build_dataset_selects_the_mid_fill_slice(self):
        trades = pd.DataFrame(
            {
                "ticker": ["A", "A"],
                "event_date": pd.to_datetime(["2020-05-01", "2020-05-01"]),
                "entry_date": pd.to_datetime(["2020-04-30", "2020-04-30"]),
                "fill_alpha": [0.0, 0.5],
                "ret": [-0.3, 0.1],
                "entry_cost": [4.0, 4.0],
                "spot_entry": [100.0, 100.0],
                "dte_entry": [2, 2],
            }
        )
        out = gate_mod.build_dataset(
            trades, panel=pd.DataFrame({"ticker": [], "date": []}), daily=pd.DataFrame(
                {"ticker": [], "date": [], "src_iv": []}
            )
        )
        assert len(out) == 1
        assert out["ret"].iloc[0] == 0.1
        assert out["entry_cost_pct"].iloc[0] == pytest.approx(4.0)
