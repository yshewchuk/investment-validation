import pytest

from engine.v2.scoring.financial import financial_diagnostics


def test_runup_surface_produces_fair_premium_and_market_ratio():
    record = {
        "spot": 100.0,
        "strike": 100.0,
        "entry_cost": 2.3,
        "driver_prediction": 6.0,
        "runup_move_prediction": 4.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {
                "intercept": 0.01,
                "implied_move": 0.004,
                "abs_moneyness": 0.003,
                "moneyness_sq_div10": 0.0,
                "signed_moneyness": 0.0,
                "implied_x_abs_moneyness_div10": 0.0,
            },
        },
    }
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(4.6)
    assert diagnostics["premium_vs_fair"] == pytest.approx(0.5)


def test_complete_constant_runup_surface_returns_two_percent():
    record = {
        "spot": 100.0,
        "strike": 100.0,
        "entry_cost": 1.0,
        "driver_prediction": 6.0,
        "runup_move_prediction": 4.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {
                "intercept": 0.02,
                "implied_move": 0.0,
                "abs_moneyness": 0.0,
                "moneyness_sq_div10": 0.0,
                "signed_moneyness": 0.0,
                "implied_x_abs_moneyness_div10": 0.0,
            },
        },
    }
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(2.0)
    assert diagnostics["premium_vs_fair"] == pytest.approx(0.5)


# -- linear-payoff arithmetic in _fair_premium (engine/v2/scoring/financial.py:72-78) --
#
# The plain-linear branch (no "kind" == "runup_payoff_surface", no model_fair_pct) computes
# fair_premium_pct = max(0.0, (intercept + slope * driver_prediction) * 100.0).
# Every expected number below is that formula worked out by hand, not the code's output.


def test_linear_payoff_general_case_intercept_plus_slope_times_driver():
    # 0.01 + 0.5 * 0.08 = 0.01 + 0.04 = 0.05 -> * 100 = 5.0
    # A max<->min floor-swap mutant would turn this positive result into 0.0.
    record = {"driver_prediction": 0.08, "payoff": {"intercept": 0.01, "slope": 0.5}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(5.0)


def test_linear_payoff_at_the_money_driver_zero_isolates_intercept():
    # 0.03 + 0.7 * 0.0 = 0.03 -> * 100 = 3.0
    # A "*" -> "+" mutant on the slope term would instead give
    # 0.03 + (0.7 + 0.0) = 0.73 -> * 100 = 73.0, so this is mutation-sensitive.
    record = {"driver_prediction": 0.0, "payoff": {"intercept": 0.03, "slope": 0.7}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(3.0)


def test_linear_payoff_deep_in_the_money_large_driver():
    # 0.005 + 2.0 * 1.25 = 0.005 + 2.5 = 2.505 -> * 100 = 250.5
    # A mutant that swaps which coefficient multiplies the driver (intercept and
    # slope interchanged) would instead give 2.0 + 0.005 * 1.25 = 2.00625 -> 200.625.
    record = {"driver_prediction": 1.25, "payoff": {"intercept": 0.005, "slope": 2.0}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(250.5)


def test_linear_payoff_deep_out_of_the_money_floors_at_zero():
    # -0.02 + 0.3 * 0.01 = -0.02 + 0.003 = -0.017 -> * 100 = -1.7 -> max(0.0, -1.7) = 0.0
    # A dropped or sign-flipped floor would surface -1.7 instead.
    record = {"driver_prediction": 0.01, "payoff": {"intercept": -0.02, "slope": 0.3}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(0.0)


def test_linear_payoff_zero_intercept_and_slope_is_exactly_zero():
    # 0.0 + 0.0 * 5.0 = 0.0 -> * 100 = 0.0 -> max(0.0, 0.0) = 0.0: the floor boundary itself.
    record = {"driver_prediction": 5.0, "payoff": {"intercept": 0.0, "slope": 0.0}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] == pytest.approx(0.0)


def test_linear_payoff_missing_slope_coefficient_is_undetermined():
    # The function documents both "intercept" and "slope" as required; missing
    # either must refuse (None), not silently treat the missing one as zero.
    record = {"driver_prediction": 0.5, "payoff": {"intercept": 0.01}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] is None


def test_linear_payoff_missing_driver_prediction_is_undetermined():
    record = {"driver_prediction": None, "payoff": {"intercept": 0.01, "slope": 0.5}}
    diagnostics = financial_diagnostics(record)
    assert diagnostics["fair_premium_pct"] is None
