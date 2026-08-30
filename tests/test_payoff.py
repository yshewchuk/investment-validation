"""The payoff calibration — the bridge from a predicted quantity to a P&L.

The two properties that matter: the fit recovers a known linear relationship,
and it never sees a trade that had not closed by the decision date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.payoff import (
    PAYOFF_DRIVER,
    PayoffError,
    PayoffMap,
    driver_for,
    fit_payoff,
)


def linear_trades(
    n: int = 500,
    *,
    intercept: float = 0.01,
    slope: float = 0.5,
    noise: float = 0.0,
    strategy: str = "STR-THRU",
    alpha: float = 0.5,
    year: int = 2020,
    seed: int = 0,
) -> pd.DataFrame:
    """Trades whose exit value follows a known line in the driver."""
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 0.15, n)  # |move| as a fraction of spot
    spot = 100.0
    exit_value = (intercept + slope * driver) * spot
    if noise:
        exit_value = exit_value + rng.normal(0, noise * spot, n)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "fill_alpha": alpha,
            "abs_move": driver,
            "im_t1": driver,
            "spot_entry": spot,
            "exit_value": exit_value,
            "exit_date": pd.Timestamp(f"{year}-06-02"),
        }
    )


class TestDriver:
    def test_through_the_print_is_driven_by_the_realized_move(self):
        assert driver_for("STR-THRU") == "abs_move"

    def test_sold_before_the_print_is_driven_by_the_implied_move(self):
        """The realized move is irrelevant to a position that is already closed."""
        assert driver_for("STR-RUNUP") == "im_t1"

    def test_cal_p_has_no_payoff_and_says_why(self):
        with pytest.raises(KeyError, match="never been backtested"):
            driver_for("CAL-P")
        assert "CAL-P" not in PAYOFF_DRIVER


class TestFit:
    def test_recovers_a_known_line(self):
        fitted = fit_payoff(linear_trades(intercept=0.02, slope=0.8), "STR-THRU", alpha=0.5)
        assert fitted.intercept == pytest.approx(0.02, abs=1e-6)
        assert fitted.slope == pytest.approx(0.8, abs=1e-6)
        assert fitted.n == 500
        assert fitted.r == pytest.approx(1.0, abs=1e-6)

    def test_reports_the_unexplained_spread(self):
        clean = fit_payoff(linear_trades(noise=0.0), "STR-THRU", alpha=0.5)
        noisy = fit_payoff(linear_trades(noise=0.02, seed=1), "STR-THRU", alpha=0.5)
        assert clean.resid_sd == pytest.approx(0.0, abs=1e-6)
        assert noisy.resid_sd > 0.01

    def test_selects_the_requested_alpha(self):
        pool = pd.concat(
            [
                linear_trades(slope=0.4, alpha=0.0),
                linear_trades(slope=0.9, alpha=0.5, seed=2),
            ],
            ignore_index=True,
        )
        assert fit_payoff(pool, "STR-THRU", alpha=0.0).slope == pytest.approx(0.4, abs=1e-6)
        assert fit_payoff(pool, "STR-THRU", alpha=0.5).slope == pytest.approx(0.9, abs=1e-6)

    def test_selects_the_requested_strategy(self):
        pool = pd.concat(
            [
                linear_trades(slope=0.4, strategy="STR-THRU"),
                linear_trades(slope=0.9, strategy="STR-RUNUP", seed=3),
            ],
            ignore_index=True,
        )
        assert fit_payoff(pool, "STR-RUNUP", alpha=0.5).slope == pytest.approx(0.9, abs=1e-6)

    def test_too_few_trades_refuses_rather_than_fitting_noise(self):
        with pytest.raises(PayoffError, match="need 200"):
            fit_payoff(linear_trades(n=20), "STR-THRU", alpha=0.5)

    def test_ignores_rows_with_a_missing_driver_or_spot(self):
        pool = linear_trades(n=400)
        pool.loc[:99, "abs_move"] = np.nan
        pool.loc[100:149, "spot_entry"] = 0.0
        fitted = fit_payoff(pool, "STR-THRU", alpha=0.5, min_trades=100)
        assert fitted.n == 250


class TestCausality:
    def test_only_trades_closed_before_the_cutoff_are_used(self):
        past = linear_trades(300, slope=0.5, year=2018)
        future = linear_trades(300, slope=5.0, year=2024, seed=9)
        pool = pd.concat([past, future], ignore_index=True)
        fitted = fit_payoff(
            pool, "STR-THRU", alpha=0.5, before=pd.Timestamp("2020-01-01")
        )
        assert fitted.n == 300
        assert fitted.slope == pytest.approx(0.5, abs=1e-6)
        assert fitted.fitted_through == pd.Timestamp("2020-01-01")

    def test_a_trade_closing_on_the_cutoff_is_excluded(self):
        pool = linear_trades(300, year=2020)  # exits 2020-06-02
        with pytest.raises(PayoffError):
            fit_payoff(pool, "STR-THRU", alpha=0.5, before=pd.Timestamp("2020-06-02"))
        assert fit_payoff(
            pool, "STR-THRU", alpha=0.5, before=pd.Timestamp("2020-06-03")
        ).n == 300


class TestResiduals:
    """The payoff's own error distribution, kept empirical rather than assumed."""

    def test_residuals_are_stored(self):
        fitted = fit_payoff(linear_trades(noise=0.02, seed=4), "STR-THRU", alpha=0.5)
        assert fitted.residuals.size == 500

    def test_draws_come_from_the_stored_distribution(self):
        fitted = fit_payoff(linear_trades(noise=0.02, seed=4), "STR-THRU", alpha=0.5)
        draws = fitted.residual_draws(1000, np.random.default_rng(0))
        assert draws.shape == (1000,)
        assert set(np.unique(draws)).issubset(set(fitted.residuals.tolist()))

    def test_draws_preserve_skew_a_gaussian_would_erase(self):
        """The reason this is empirical rather than a normal of the same sd."""
        rng = np.random.default_rng(11)
        n = 4000
        driver = rng.uniform(0.0, 0.15, n)
        spot = 100.0
        # A long-vol shape: mostly small negative residuals, occasionally large
        # positive ones.
        skewed = rng.exponential(0.02, n) - 0.015
        frame = pd.DataFrame(
            {
                "strategy": "STR-THRU", "fill_alpha": 0.5, "abs_move": driver,
                "spot_entry": spot,
                "exit_value": (0.01 + 0.5 * driver + skewed) * spot,
                "exit_date": pd.Timestamp("2020-06-02"),
            }
        )
        fitted = fit_payoff(frame, "STR-THRU", alpha=0.5)
        draws = fitted.residual_draws(20000, np.random.default_rng(1))
        # Median below mean is the signature of right skew; a Gaussian would put
        # them on top of each other.
        assert np.median(draws) < np.mean(draws) - 1e-4
        assert (draws < 0).mean() > 0.55

    def test_the_subsample_is_capped_and_deterministic(self):
        from engine.payoff import MAX_RESIDUALS

        big = linear_trades(MAX_RESIDUALS + 2000, noise=0.02, seed=6)
        a = fit_payoff(big, "STR-THRU", alpha=0.5)
        b = fit_payoff(big, "STR-THRU", alpha=0.5)
        assert a.residuals.size == MAX_RESIDUALS
        assert np.array_equal(a.residuals, b.residuals)

    def test_falls_back_to_a_gaussian_without_residuals(self):
        fitted = PayoffMap(
            strategy="STR-THRU", driver="abs_move", alpha=0.5, intercept=0.0,
            slope=1.0, resid_sd=0.05, n=500, r=1.0,
        )
        draws = fitted.residual_draws(5000, np.random.default_rng(0))
        assert abs(float(np.std(draws)) - 0.05) < 0.01

    def test_a_degenerate_fit_draws_zeros(self):
        fitted = PayoffMap(
            strategy="STR-THRU", driver="abs_move", alpha=0.5, intercept=0.0,
            slope=1.0, resid_sd=0.0, n=500, r=1.0,
        )
        assert not fitted.residual_draws(10, np.random.default_rng(0)).any()


class TestApplication:
    def map(self, **kwargs) -> PayoffMap:
        base = dict(
            strategy="STR-THRU", driver="abs_move", alpha=0.5,
            intercept=0.0, slope=1.0, resid_sd=0.0, n=500, r=1.0,
        )
        base.update(kwargs)
        return PayoffMap(**base)

    def test_exit_value_scales_with_spot(self):
        assert self.map().exit_value([0.05], 200.0)[0] == pytest.approx(10.0)

    def test_exit_value_is_floored_at_zero(self):
        """A long structure cannot be worth less than nothing; a line can say so."""
        fitted = self.map(intercept=-0.05, slope=1.0)
        assert fitted.exit_value([0.0], 100.0)[0] == 0.0

    def test_pnl_and_return_use_the_real_entry_cost(self):
        fitted = self.map()
        assert fitted.pnl([0.05], 100.0, 4.0)[0] == pytest.approx(1.0)
        assert fitted.ret([0.05], 100.0, 4.0)[0] == pytest.approx(0.25)

    def test_return_on_a_non_positive_cost_is_nan(self):
        assert np.isnan(self.map().ret([0.05], 100.0, 0.0)[0])

    def test_serializes_with_its_cutoff(self):
        doc = self.map(fitted_through=pd.Timestamp("2021-01-01")).as_dict()
        assert doc["fitted_through"] == "2021-01-01"
        assert doc["slope"] == 1.0
