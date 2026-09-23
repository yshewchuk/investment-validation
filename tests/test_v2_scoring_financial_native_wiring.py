"""Regression test: native's own stage outputs feed financial_diagnostics.

``engine/v2/scoring/application.py`` calls
``financial.financial_diagnostics(values)`` on whatever ``values`` the
scoring pipeline assembled. That function needs four facts it never
computes itself -- ``payoff``, ``implied_move``, ``driver_name`` and
``strike`` -- and expects them under exactly legacy's field names
(``engine/score.py``'s ``ScoreResult`` fields, ``financial.py:33-78``).

Before this fix, no native stage ever published ``payoff`` or ``strike``
into ``values``: the model stage (``_execute_model``/``_execute_runup_model``,
``stages.py``) fit the payoff line/surface and used it to simulate
``exp_pnl_model``/``win_model``, then discarded the fit as a local variable;
the pricing stage (``_publish_pricing``) never surfaced the resolved
strike at all. ``fair_premium_pct``/``model_vs_market``/``premium_vs_fair``
came out ``None`` on every driver-role row as a result.

These tests build a real ``SourceBundle``, run it through the actual
``build_native_score_inputs`` -> ``assemble_native_values`` pipeline (no
mocking of the stages under test), and check that ``financial_diagnostics``
on the assembled ``values`` produces the same numbers an independent,
by-hand reimplementation of legacy's formula would -- not merely
"non-None". The independent formula is transcribed from
``engine/v2/scoring/financial.py`` (the module under test is not imported
twice; the arithmetic is duplicated so a shared bug would not hide behind
a shared implementation).
"""
from __future__ import annotations

from math import exp, log

import pytest

from engine.v2.scoring.financial import financial_diagnostics
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import assemble_native_values

ORATS_EMOVE_FACTOR = 0.645


def _legacy_model_vs_market(driver_name, driver, implied):
    if driver_name != "abs_move" or driver is None or implied in (None, 0, 0.0):
        return None
    return float(driver) / (float(implied) * ORATS_EMOVE_FACTOR)


def _legacy_line_fair_premium(payoff, driver):
    intercept, slope = payoff.get("intercept"), payoff.get("slope")
    if driver is None or intercept is None or slope is None:
        return None
    return max(0.0, (float(intercept) + float(slope) * float(driver)) * 100.0)


def _legacy_surface_fair_premium(payoff, driver, move, spot, strike):
    coefficients = payoff.get("coefficients") or {}
    names = ("intercept", "implied_move", "abs_moneyness", "moneyness_sq_div10",
              "signed_moneyness", "implied_x_abs_moneyness_div10")
    if (driver is None or move is None or spot in (None, 0, 0.0)
            or strike in (None, 0, 0.0) or any(name not in coefficients for name in names)):
        return None
    values = []
    for direction in (-1.0, 1.0):
        exit_spot = float(spot) * exp(direction * float(move) / 100.0)
        money = 100.0 * log(exit_spot / float(strike))
        absolute = abs(money)
        terms = (1.0, float(driver), absolute, money * money / 10.0,
                 money, float(driver) * absolute / 10.0)
        values.append(sum(float(coefficients[name]) * term
                          for name, term in zip(names, terms)))
    return max(0.0, sum(values) / len(values) * 100.0)


def test_str_thru_line_payoff_is_wired_from_native_stages():
    """STR-THRU: the model stage's fitted line reaches financial_diagnostics.

    ``payoff_source_rows`` are two prior trades chosen so the causal OLS fit
    through them is exact (a line through two points has zero residual at
    both): driver=0 -> exit_value/spot=0.02, driver=10 -> exit_value/spot=0.06,
    giving slope=0.004, intercept=0.02 by hand.
    """
    expiry = "2026-09-18"
    strike = 100.0
    quotes = {
        ("C", strike, expiry): {"bid": 2.0, "ask": 3.0},
        ("P", strike, expiry): {"bid": 2.0, "ask": 3.0},
    }
    bundle = SourceBundle(
        source_ref="wiring-str-thru",
        context={
            "ticker": "WIRE", "event_date": "2026-09-16",
            "entry_date": "2026-09-16", "exit_date": "2026-09-17",
            "expiry": expiry, "spot": 100.0,
            # A raw, source-owned quote fact (today's market implied move at
            # decision time) -- never a caller-pinned strike, so ``strike``
            # below can only come from geometry's own resolution.
            "implied_move": 6.0,
        },
        raw_quotes=quotes,
        feature_vector={"zero": 0.0},
        feature_missing_mask={},
        model_identity={"driver": {"model_id": "wiring-driver-v1"}},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:wiring-driver"},
        residual_recipe={
            "terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        analog_recipe={},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
        payoff_recipe={"min_trades": 2, "seed": 20260922, "draw_count": 8},
        payoff_source_rows=[
            {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0,
             "exit_date": "2026-09-10"},
            {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0,
             "exit_date": "2026-09-10"},
        ],
        model_residual_rows=[{"prediction": 7.0, "residual": 0.0}],
    )
    inputs = build_native_score_inputs(bundle)
    values = assemble_native_values(inputs, strategy="STR-THRU")

    assert values["flags"] == ()
    assert values["strike"] == 100.0  # resolved by geometry, not pinned in context
    assert values["implied_move"] == 6.0
    assert values["driver_name"] == "abs_move"
    assert values["payoff"]["intercept"] == pytest.approx(0.02)
    assert values["payoff"]["slope"] == pytest.approx(0.004)

    diagnostics = financial_diagnostics(values)

    expected_fair = _legacy_line_fair_premium(values["payoff"], values["driver_prediction"])
    expected_model_vs_market = _legacy_model_vs_market(
        values["driver_name"], values["driver_prediction"], values["implied_move"],
    )
    assert diagnostics["fair_premium_pct"] is not None
    assert diagnostics["model_vs_market"] is not None
    assert diagnostics["fair_premium_pct"] == pytest.approx(4.8)
    assert diagnostics["fair_premium_pct"] == expected_fair
    assert diagnostics["model_vs_market"] == expected_model_vs_market


def test_str_runup_surface_payoff_is_wired_from_native_stages():
    """STR-RUNUP: the model stage's fitted surface reaches financial_diagnostics.

    ``driver_name`` is set to "abs_move" here (legacy's real STR-RUNUP rows
    carry "im_t1", engine/score.py:3032) purely so this test can exercise
    model_vs_market's wiring too -- that field only ever fires for
    driver_name == "abs_move" (financial.py:36), a strategy-independent
    rule this test is checking, not STR-RUNUP's actual production label.
    """
    expiry = "2026-09-30"
    strike = 100.0
    quotes = {
        ("C", strike, expiry): {"bid": 2.0, "ask": 3.0},
        ("P", strike, expiry): {"bid": 2.0, "ask": 3.0},
    }
    bundle = SourceBundle(
        source_ref="wiring-str-runup",
        context={
            "ticker": "WIRE", "event_date": "2026-09-30",
            "entry_date": "2026-09-16", "exit_date": "2026-09-17",
            "expiry": expiry, "spot": 100.0,
            "implied_move": 6.0,
        },
        raw_quotes=quotes,
        feature_vector={"days_before_print": 10.0},
        feature_missing_mask={},
        model_identity={"driver": {"model_id": "wiring-implied-v1"},
                        "runup_move": {"model_id": "wiring-move-v1"}},
        forecast_recipes={
            "driver_prediction": {"intercept": 5.0, "coefficients": {}},
            "runup_move_prediction": {"intercept": 8.0, "coefficients": {}},
        },
        model_artifact_refs={
            "driver_prediction": "sha256:wiring-implied",
            "runup_move_prediction": "sha256:wiring-move",
        },
        residual_recipe={
            "terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        analog_recipe={},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
        driver_name="abs_move",
        strategy="STR-RUNUP",
        payoff_recipe={"min_trades": 2, "seed": 20260922, "draw_count": 8},
        payoff_source_rows=[
            {"driver": 0.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
             "exit_value": 2.0, "exit_date": "2026-09-01"},
            {"driver": 5.0, "spot_entry": 100.0, "spot_exit": 110.0, "strike": 100.0,
             "exit_value": 6.0, "exit_date": "2026-09-01"},
            {"driver": 8.0, "spot_entry": 100.0, "spot_exit": 90.0, "strike": 100.0,
             "exit_value": 5.0, "exit_date": "2026-09-01"},
        ],
        model_residual_rows=[{"prediction": 5.0, "residual": 0.0}],
        runup_move_residual_rows=[{"prediction": 8.0, "residual": 0.0}],
    )
    inputs = build_native_score_inputs(bundle)
    values = assemble_native_values(inputs, strategy="STR-RUNUP")

    assert values["flags"] == ()
    assert values["strike"] == 100.0
    assert values["implied_move"] == 6.0
    assert values["driver_name"] == "abs_move"
    assert values["payoff"]["kind"] == "runup_payoff_surface"
    assert set(values["payoff"]["coefficients"]) == {
        "intercept", "implied_move", "abs_moneyness", "moneyness_sq_div10",
        "signed_moneyness", "implied_x_abs_moneyness_div10",
    }

    diagnostics = financial_diagnostics(values)

    expected_fair = _legacy_surface_fair_premium(
        values["payoff"], values["driver_prediction"],
        values["runup_move_prediction"], values["spot"], values["strike"],
    )
    expected_model_vs_market = _legacy_model_vs_market(
        values["driver_name"], values["driver_prediction"], values["implied_move"],
    )
    assert diagnostics["fair_premium_pct"] is not None
    assert diagnostics["model_vs_market"] is not None
    assert diagnostics["fair_premium_pct"] == expected_fair
    assert diagnostics["model_vs_market"] == expected_model_vs_market
