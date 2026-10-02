"""Planned-exit simulation values each leg by its own right (issue #170).

The simulation used to price every leg with the put formula and ignore
``leg.right``. These tests pin call and put legs against independent
references: hand-computed intrinsic values at expiry, a closed-form
Black-Scholes call written here (not the production put-call-parity code), and
put-call parity itself.
"""
import math

import numpy as np
import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.domain.generation.structures import PricedLeg, Pricing
from engine.v2.scoring import stages
from tests.planned_exit_support import legacy_expected_pnl

EVENT = "2026-09-16"


def _block(dte_exit: float, pre_iv30: float = 40.0) -> dict:
    # Zero residuals: the exit spot is the entry spot and the exit vol is
    # pre_iv30, so every simulated draw is the same deterministic scenario.
    rows = [
        {"event_date": str(day.date()), "pred_abs_move": 0.0,
         "err_move": 0.0, "err_crush": 0.0}
        for day in pd.date_range("2024-01-01", periods=300, freq="D")
    ]
    return {"mode": "planned_exit", "pre_iv30": pre_iv30, "dte_exit": dte_exit,
            "event_date": EVENT, "residuals": rows}


def _leg(right: str, side: str = "buy", strike: float = 100.0, qty: float = 1.0):
    return PricedLeg("leg", right, side, qty, strike, "2026-09-18",
                     0.0, 0.0, 0.0, 0.0)


def _simulate(legs, *, spot: float, cost: float, dte_exit: float = 0.0):
    values = {"spot": spot, "entry_cost": cost, "forecast_abs_move": 0.0,
              "pred_iv_crush": 0.0}
    pricing = Pricing("STR-TEST", spot, cost, tuple(legs))
    flags: list[str] = []
    output = stages._planned_exit_simulation(
        _block(dte_exit), values, pricing, "STR-TEST", flags,
    )
    return output, flags


@pytest.mark.parametrize(("right", "side", "spot", "intrinsic"), [
    ("C", "buy", 120.0, 20.0),
    ("C", "buy", 80.0, 0.0),
    ("P", "buy", 120.0, 0.0),
    ("P", "buy", 80.0, 20.0),
    ("C", "sell", 120.0, -20.0),
    ("P", "sell", 80.0, -20.0),
])
def test_expiry_value_is_the_legs_own_intrinsic(right, side, spot, intrinsic):
    output, flags = _simulate([_leg(right, side)], spot=spot, cost=5.0)

    assert flags == []
    assert output["exp_pnl_sim"] == pytest.approx((intrinsic - 5.0) / 5.0)


def test_issue_reproduction_long_call_costing_five_at_zero_time():
    """spot 120, strike 100, call paid 5: +300%, not the put's -100%."""
    call, _ = _simulate([_leg("C")], spot=120.0, cost=5.0)
    put, _ = _simulate([_leg("P")], spot=120.0, cost=5.0)

    assert call["exp_pnl_sim"] == pytest.approx(3.0)
    assert put["exp_pnl_sim"] == pytest.approx(-1.0)


def _closed_form_call(spot: float, strike: float, years: float, vol: float) -> float:
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
    return spot * cdf(d1) - strike * cdf(d2)


@pytest.mark.parametrize(("spot", "strike"), [(100.0, 100.0), (100.0, 90.0), (100.0, 115.0)])
def test_pre_expiry_call_matches_closed_form_and_parity_holds(spot, strike):
    dte, vol, cost = 9.0, 0.40, 3.0
    call, _ = _simulate([_leg("C", strike=strike)], spot=spot, cost=cost, dte_exit=dte)
    put, _ = _simulate([_leg("P", strike=strike)], spot=spot, cost=cost, dte_exit=dte)
    call_value = call["exp_pnl_sim"] * cost + cost
    put_value = put["exp_pnl_sim"] * cost + cost

    assert call_value == pytest.approx(
        _closed_form_call(spot, strike, dte / 365.0, vol), rel=1e-9,
    )
    assert call_value - put_value == pytest.approx(spot - strike, abs=1e-9)


def test_unrecognized_right_refuses_without_numbers():
    output, flags = _simulate([_leg("X")], spot=120.0, cost=5.0)

    assert output == {}
    assert flags == ["UNSUPPORTED_SIMULATION_LEG:right"]


def test_straddle_is_the_sum_of_its_call_and_put():
    both, _ = _simulate([_leg("C"), _leg("P")], spot=120.0, cost=5.0)

    assert both["exp_pnl_sim"] == pytest.approx((20.0 - 5.0) / 5.0)


def test_reference_helper_equals_legacy_expected_pnl_on_put_only_legs():
    """Anchors the right-aware parity reference to the legacy kernel."""
    rng = np.random.default_rng(7)
    history = pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=300, freq="D"),
        "pred_abs_move": 3.0 + rng.random(300) * 5.0,
        "err_move": rng.normal(0.0, 2.0, 300),
        "err_crush": rng.normal(-5.0, 10.0, 300),
    })
    legs = [{"strike": 100.0, "qty": 1.0, "side": "sell"},
            {"strike": 95.0, "qty": 1.0, "side": "buy"}]
    kwargs = dict(spot=100.0, entry_cost=1.5, pre_iv30=40.0, pred_abs_move=6.0,
                  pred_iv_crush=-20.0, dte_exit=9.0,
                  event_date=pd.Timestamp(EVENT).normalize(), key="STR-TEST")

    legacy = pnl_sim.expected_pnl(
        exit_legs=legs, pool=pnl_sim.ResidualPool(history), **kwargs)
    reference = legacy_expected_pnl(
        exit_legs=legs, pool=pnl_sim.ResidualPool(history), **kwargs)

    assert reference == legacy
