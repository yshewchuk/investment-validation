"""Calibration measurement.

Checked against constructions whose right answer is known by hand: a perfectly
calibrated forecaster, a systematically overconfident one, and a useless one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import calibrate as cal


class TestBrier:
    def test_perfect_forecast_scores_zero(self):
        assert cal.brier([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_maximally_wrong_forecast_scores_one(self):
        assert cal.brier([0.0, 1.0], [1, 0]) == 1.0

    def test_ignores_non_finite(self):
        assert cal.brier([1.0, np.nan], [1, 0]) == 0.0

    def test_empty_is_nan_not_zero(self):
        assert np.isnan(cal.brier([], []))


class TestBrierSkill:
    def test_the_base_rate_predictor_scores_zero_skill(self):
        outcome = np.array([1, 1, 0, 0, 1, 0, 1, 0.0])
        base = np.full(len(outcome), outcome.mean())
        assert cal.brier_skill(base, outcome) == pytest.approx(0.0)

    def test_a_perfect_forecast_scores_one(self):
        outcome = np.array([1, 0, 1, 0.0])
        assert cal.brier_skill(outcome, outcome) == pytest.approx(1.0)

    def test_worse_than_the_base_rate_is_negative(self):
        outcome = np.array([1, 1, 1, 0.0])
        assert cal.brier_skill(1 - outcome, outcome) < 0

    def test_a_constant_outcome_has_no_defined_skill(self):
        assert np.isnan(cal.brier_skill([0.5, 0.5], [1, 1]))


class TestReliability:
    def test_a_calibrated_forecaster_tracks_the_diagonal(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0.05, 0.95, 4000)
        outcome = (rng.uniform(size=4000) < prob).astype(float)
        table = cal.reliability_table(prob, outcome)
        assert len(table) == 10
        assert (table["gap"].abs() < 0.06).all()

    def test_an_overconfident_forecaster_shows_a_negative_gap(self):
        rng = np.random.default_rng(1)
        truth = rng.uniform(0.1, 0.6, 4000)
        outcome = (rng.uniform(size=4000) < truth).astype(float)
        table = cal.reliability_table(truth + 0.3, outcome)
        assert table["gap"].mean() < -0.2

    def test_buckets_hold_equal_counts(self):
        table = cal.reliability_table(np.linspace(0, 1, 1000), np.zeros(1000))
        assert table["n"].nunique() == 1

    def test_bins_shrink_on_a_small_sample(self):
        table = cal.reliability_table(np.linspace(0, 1, 50), np.zeros(50))
        assert len(table) < 10

    def test_empty_input_returns_the_empty_shape(self):
        table = cal.reliability_table([], [])
        assert table.empty and "predicted" in table.columns


class TestDeciles:
    def test_orders_by_prediction(self):
        predicted = np.linspace(-1, 1, 500)
        table = cal.decile_table(predicted, predicted)
        assert table["predicted"].is_monotonic_increasing
        assert table["realized"].is_monotonic_increasing

    def test_a_uniform_bias_shows_in_the_gap_not_the_ordering(self):
        predicted = np.linspace(-1, 1, 500)
        table = cal.decile_table(predicted, predicted - 0.5)
        assert table["gap"].mean() == pytest.approx(-0.5, abs=1e-9)
        assert cal.monotonicity(table) == pytest.approx(1.0)


class TestMonotonicity:
    def test_perfect_ordering_is_one(self):
        table = pd.DataFrame({"realized": [0.1, 0.2, 0.3, 0.4]})
        assert cal.monotonicity(table) == pytest.approx(1.0)

    def test_reversed_ordering_is_minus_one(self):
        table = pd.DataFrame({"realized": [0.4, 0.3, 0.2, 0.1]})
        assert cal.monotonicity(table) == pytest.approx(-1.0)

    def test_a_flat_column_has_no_defined_ordering(self):
        table = pd.DataFrame({"realized": [0.2, 0.2, 0.2, 0.2]})
        assert np.isnan(cal.monotonicity(table))

    def test_too_few_buckets_is_nan(self):
        assert np.isnan(cal.monotonicity(pd.DataFrame({"realized": [0.1, 0.2]})))


class TestCalibrate:
    def frame(self, n=2000, *, shift=0.0, seed=0):
        rng = np.random.default_rng(seed)
        prob = rng.uniform(0.1, 0.9, n)
        wins = rng.uniform(size=n) < prob
        return pd.DataFrame(
            {
                "win_model": prob + shift,
                "exp_pnl_model": prob - 0.5,
                "ret": np.where(wins, 0.5, -0.4),
                "year": rng.integers(2018, 2025, n),
            }
        )

    def test_a_calibrated_model_beats_the_base_rate(self):
        report = cal.calibrate(self.frame(), label="test")
        assert report.beats_base_rate
        assert report.brier_skill > 0.1
        assert report.reliability_monotonicity > 0.9

    def test_an_overconfident_model_still_ranks_but_misses_the_level(self):
        report = cal.calibrate(self.frame(shift=0.25), label="over")
        assert report.reliability_monotonicity > 0.9
        assert report.reliability["gap"].mean() < -0.15

    def test_a_useless_model_does_not_beat_the_base_rate(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "win_model": rng.uniform(0.1, 0.9, 2000),
                "exp_pnl_model": rng.normal(0, 1, 2000),
                "ret": rng.normal(0, 1, 2000),
                "year": 2020,
            }
        )
        report = cal.calibrate(frame, label="noise")
        assert report.brier_skill < 0.05

    def test_reports_years_covered(self):
        report = cal.calibrate(self.frame(), label="test")
        assert report.years and min(report.years) >= 2018

    def test_a_small_sample_is_flagged_not_silently_reported(self):
        report = cal.calibrate(self.frame(n=50), label="small")
        assert any("indicative" in note for note in report.notes)

    def test_serializes_for_a_report(self):
        doc = cal.calibrate(self.frame(), label="test").as_dict()
        assert doc["label"] == "test"
        assert len(doc["reliability"]) == 10
        assert "beats_base_rate" in doc
