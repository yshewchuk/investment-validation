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
