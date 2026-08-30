"""The win-rate plumbing, proved against a strategy whose truth is known.

The real calibration check grades the scorer against *actual* outcomes, so when
it fails we cannot tell whether the probability machinery is wrong or the models
are simply young. This test removes that ambiguity: it builds a mocked strategy
with a known data-generating process, runs the exact payoff/draw pipeline the
scorer uses (:func:`engine.payoff.fit_payoff` → empirical residual draws →
:func:`engine.payoff.simulate_returns`), and requires it to recover the known win
rate. If this passes, a real-data calibration failure is attributable to the
data/models, not to the plumbing.

The mocked strategy is deliberately long-vol-shaped: right-skewed exit residuals
(many small losses, occasional large gains). That is the case a naive Gaussian
noise draw gets wrong in the over-optimistic direction, so recovering the truth
here is the meaningful bar.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.payoff import fit_payoff, simulate_returns

# --- the known data-generating process ------------------------------------
#
# exit_value / spot = INTERCEPT + SLOPE * driver + eps, with a right-skewed eps.
# A trade wins when exit_value > cost. Because eps is exponential (shifted), the
# win probability at any driver point is available in closed form — that is the
# "known truth" the pipeline must reproduce.
INTERCEPT = 0.01
SLOPE = 0.5
EPS_SCALE = 0.02
EPS_SHIFT = 0.015
COST_PER_SPOT = 0.05


def true_win(driver: float, cost_per_spot: float = COST_PER_SPOT) -> float:
    """Closed-form P(exit_value > cost) at a given driver value."""
    t = cost_per_spot - INTERCEPT - SLOPE * driver
    z = t + EPS_SHIFT
    return 1.0 if z <= 0 else float(np.exp(-z / EPS_SCALE))


def mocked_trades(n: int = 4000, *, seed: int = 0, spot: float = 100.0):
    """n trades from the mocked strategy, in the shape fit_payoff consumes."""
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 0.15, n)
    eps = rng.exponential(EPS_SCALE, n) - EPS_SHIFT
    exit_value = (INTERCEPT + SLOPE * driver + eps) * spot
    return {
        "driver": driver,
        "exit_value": exit_value,
        "spot": spot,
        "cost": COST_PER_SPOT * spot,
    }


def pipeline_win(fitted, driver_point: float, spot: float, cost: float, rng) -> float:
    """The scorer's win computation, at one driver point with no model error."""
    draws = np.full(4000, driver_point)
    noise = fitted.residual_draws(4000, rng)
    returns = simulate_returns(draws, fitted, spot, cost, noise)
    return float((returns > 0).mean())


@pytest.fixture(scope="module")
def fitted():
    m = mocked_trades()
    import pandas as pd

    frame = pd.DataFrame(
        {
            "strategy": "STR-THRU",
            "fill_alpha": 0.5,
            "abs_move": m["driver"],
            "spot_entry": m["spot"],
            "exit_value": m["exit_value"],
            "exit_date": pd.Timestamp("2020-06-02"),
        }
    )
    return fit_payoff(frame, "STR-THRU", alpha=0.5)


class TestWinPlumbing:
    def test_recovers_the_payoff_line_of_the_mock(self, fitted):
        """The fit must recover the mock's mean exit line (eps mean absorbed)."""
        eps_mean = EPS_SCALE - EPS_SHIFT
        assert fitted.slope == pytest.approx(SLOPE, abs=0.02)
        assert fitted.intercept == pytest.approx(INTERCEPT + eps_mean, abs=0.01)

    @pytest.mark.parametrize("driver_point", [0.03, 0.07, 0.10])
    def test_recovers_a_known_win_rate(self, fitted, driver_point):
        """The headline mechanics assertion: pipeline win == closed-form truth."""
        m = mocked_trades()
        got = pipeline_win(
            fitted, driver_point, m["spot"], m["cost"], np.random.default_rng(1)
        )
        want = true_win(driver_point)
        assert got == pytest.approx(want, abs=0.04), (
            f"at driver {driver_point}: plumbing says {got:.3f}, truth is {want:.3f}"
        )

    def test_win_rate_is_monotone_in_the_driver(self, fitted):
        """A bigger predicted move must never produce a lower win rate."""
        m = mocked_trades()
        rng = np.random.default_rng(2)
        wins = [pipeline_win(fitted, d, m["spot"], m["cost"], rng) for d in
                (0.02, 0.05, 0.08, 0.11)]
        assert all(b >= a - 0.02 for a, b in zip(wins, wins[1:])), wins

    def test_empirical_residuals_avoid_the_gaussian_overstatement(self, fitted):
        """Why the noise is empirical, not a normal of the same sd.

        A Gaussian noise draw symmetric about the line puts too much mass above
        it for a right-skewed payoff, overstating P(profit). The empirical draw
        must not: its win estimate should sit *below* the Gaussian-noise estimate
        and track the truth.
        """
        m = mocked_trades()
        driver_point = 0.05
        truth = true_win(driver_point)
        empirical = pipeline_win(
            fitted, driver_point, m["spot"], m["cost"], np.random.default_rng(3)
        )
        gaussian_noise = np.random.default_rng(3).normal(0.0, fitted.resid_sd, 4000)
        draws = np.full(4000, driver_point)
        gaussian = simulate_returns(draws, fitted, m["spot"], m["cost"], gaussian_noise)
        gaussian_win = float((gaussian > 0).mean())
        assert empirical == pytest.approx(truth, abs=0.04)
        assert gaussian_win > empirical, (
            f"Gaussian noise {gaussian_win:.3f} should overstate vs empirical "
            f"{empirical:.3f} for a right-skewed payoff"
        )

    def test_no_noise_recovers_the_deterministic_line(self, fitted):
        """With zero residuals the win is a step at the break-even driver."""
        m = mocked_trades()
        # break-even driver: line(driver) == cost  ->  driver
        be = (COST_PER_SPOT - fitted.intercept) / fitted.slope
        below = simulate_returns(np.full(10, be - 0.02), fitted, m["spot"], m["cost"])
        above = simulate_returns(np.full(10, be + 0.02), fitted, m["spot"], m["cost"])
        assert (below > 0).mean() == 0.0
        assert (above > 0).mean() == 1.0
