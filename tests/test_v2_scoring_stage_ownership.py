from dataclasses import replace

import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.scoring import application
from engine.v2.scoring.native_analog import source_population_hash
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
)


def _request(alpha: float = 0.5) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-owned",
        calendar_revision="cal-1",
        strategy_version="STR-THRU",
        deployment_id="dep-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snap-1",
        mode="replay",
        fill_model={"alpha": alpha},
    )


def _native() -> NativeScoreInputs:
    context = {
        "ticker": "AAA",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-16",
        "exit_date": "2026-09-18",
        "expiry": "2026-09-18",
        "spot": 100.0,
    }
    forecast = {
        "driver_name": "abs_move",
        "models": {
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "forecast_abs_move": {"intercept": 7.0, "coefficients": {}},
        },
    }
    geometry = generate(
        "STR-THRU", {**context, "forecast_abs_move": 7.0},
    )
    quotes = {
        (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 3.0}
        for leg in geometry.legs
    }
    quote_carrier = price(geometry, quotes, 0.5)
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast=forecast,
        geometry=geometry,
        pricing=quote_carrier,
        analogs={},
        simulation={"terminal_spots": (100.0, 110.0)},
        gate={
            "model": {
                "intercept": 0.0,
                "coefficients": {"exp_pnl_sim": 1.0},
            },
            "threshold": 0.0,
        },
        chooser={},
        diagnostics={
            "forecast_abs_move": 999.0,
            "exp_pnl_sim": 999.0,
            "gate_pass": False,
            "entry_cost": 999.0,
            "model_fair_pct": 999.0,
        },
        source_ref="typed-native-fixture",
        stage_receipts=receipts,
    )


def test_fill_recomputes_owned_simulation_and_gate_outputs():
    inputs = _native()
    worst = application.score_one(_request(0.0), inputs)
    best = application.score_one(_request(1.0), inputs)

    assert worst.financial_diagnostics["entry_cost_pct"] == pytest.approx(6.0)
    assert best.financial_diagnostics["entry_cost_pct"] == pytest.approx(2.0)
    assert worst.forecasts["exp_pnl_sim"] == pytest.approx(-1.0 / 6.0)
    assert best.forecasts["exp_pnl_sim"] == pytest.approx(1.5)
    assert worst.gate_terms["gate_pass"] is False
    assert best.gate_terms["gate_pass"] is True
    assert worst.forecasts["forecast_abs_move"] == pytest.approx(7.0)
    assert all(
        item["output_hash"] != "declared-output"
        for item in worst.resolved_request["native_stage_receipts"]
    )


def test_empty_compatibility_input_refuses_missing_essentials():
    record = application.score_one(
        _request(), NativeScoreInputs.from_legacy_fields({}),
    )

    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert "MISSING_SPOT" in record.reason_codes
    assert record.financial_diagnostics["entry_cost_pct"] is None


def _residuals() -> list[dict]:
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    return [
        {
            "event_date": str(day.date()),
            "pred_abs_move": 4.0 + index % 20 / 10.0,
            "err_move": (index % 11 - 5) / 4.0,
            "err_crush": float(index % 9 - 4),
        }
        for index, day in enumerate(dates)
    ]


def _planned_native() -> NativeScoreInputs:
    base = _native()
    context = {
        **base.context,
        "event_date": "2026-09-16",
        "exit_date": "2026-09-09",
    }
    forecast = {
        **base.forecast,
        "models": {
            **base.forecast["models"],
            "pred_iv_crush": {"intercept": -20.0, "coefficients": {}},
            "model_fair_pct": {"intercept": 4.6, "coefficients": {}},
        },
    }
    return replace(
        base,
        context=context,
        forecast=forecast,
        simulation={
            "mode": "planned_exit",
            "pre_iv30": 40.0,
            "residuals": _residuals(),
        },
    )


def test_planned_exit_simulation_matches_legacy_kernel_and_is_deterministic():
    inputs = _planned_native()
    request = _request(0.5)
    first = application.score_one(request, inputs)
    second = application.score_one(request, inputs)
    priced = price(
        inputs.geometry,
        {
            (leg.right, leg.strike, leg.expiry): {
                "bid": leg.bid,
                "ask": leg.ask,
            }
            for leg in inputs.pricing.legs
        },
        0.5,
    )
    history = pd.DataFrame(_residuals())
    expected = pnl_sim.expected_pnl(
        exit_legs=[
            {
                "strike": leg.strike,
                "qty": leg.quantity,
                "side": "sell" if leg.side == "buy" else "buy",
            }
            for leg in priced.legs
        ],
        spot=100.0,
        entry_cost=priced.entry_cost,
        pre_iv30=40.0,
        pred_abs_move=7.0,
        pred_iv_crush=-20.0,
        dte_exit=9.0,
        event_date="2026-09-16",
        pool=pnl_sim.ResidualPool(history),
        key="STR-THRU",
    )

    assert first.forecasts["exp_pnl_sim"] == pytest.approx(
        expected["exp_pnl_sim"],
    )
    assert first.resolved_request["win_sim"] == pytest.approx(
        expected["win_sim"],
    )
    assert first.resolved_request["sim_p10"] == pytest.approx(
        expected["sim_p10"],
    )
    assert first.resolved_request["sim_p90"] == pytest.approx(
        expected["sim_p90"],
    )
    assert first.resolved_request["pool_n"] == expected["pool_n"]
    assert first.forecasts == second.forecasts
    assert first.resolved_request["sim_p10"] == second.resolved_request["sim_p10"]


def test_planned_exit_missing_inputs_refuse_without_fabricated_outputs():
    inputs = _planned_native()
    inputs = replace(
        inputs,
        simulation={"mode": "planned_exit", "pre_iv30": 40.0},
    )

    record = application.score_one(_request(), inputs)

    assert record.validation_status == "refused"
    assert "MISSING_SIMULATION_INPUT:residuals" in record.reason_codes
    assert record.forecasts["exp_pnl_sim"] is None


def test_driver_name_without_driver_output_refuses():
    inputs = replace(
        _native(),
        forecast={"driver_name": "abs_move"},
    )

    record = application.score_one(_request(), inputs)

    assert record.validation_status == "refused"
    assert "MISSING_FORECAST_OUTPUT:driver" in record.reason_codes
    assert record.forecasts["driver_prediction"] is None


class _ExecutorRefusal(ValueError):
    reason_codes = ("MISSING_FEATURES",)


class _RefusingExecutor:
    def predict(self, features):
        raise _ExecutorRefusal("missing alpha")


class _NonfiniteExecutor:
    def predict(self, features):
        return {"driver_prediction": float("nan")}


class _GateExecutor:
    def predict(self, features):
        return {"gate_score": features["entry_cost"]}


def test_executor_refusal_preserves_specific_reason_code():
    inputs = replace(
        _native(),
        forecast={
            "driver_name": "abs_move",
            "required_roles": ("driver",),
            "executors": {"driver_prediction": _RefusingExecutor()},
        },
    )

    record = application.score_one(_request(), inputs)

    assert "MISSING_FEATURES" in record.reason_codes
    assert "INVALID_FORECAST_EXECUTOR:driver_prediction" not in record.reason_codes


def test_nonfinite_executor_output_is_not_misreported_as_missing():
    inputs = replace(
        _native(),
        forecast={
            "driver_name": "abs_move",
            "required_roles": ("driver",),
            "executors": {"driver_prediction": _NonfiniteExecutor()},
        },
    )

    record = application.score_one(_request(), inputs)

    assert "NONFINITE_FORECAST_OUTPUT:driver_prediction" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:driver" not in record.reason_codes


def test_gate_receipt_hash_excludes_runtime_executor_identity():
    first = replace(
        _native(),
        gate={"threshold": 0.0, "executors": {"gate_score": _GateExecutor()}},
    )
    second = replace(
        _native(),
        gate={"threshold": 0.0, "executors": {"gate_score": _GateExecutor()}},
    )

    first_record = application.score_one(_request(), first)
    second_record = application.score_one(_request(), second)
    first_gate = next(
        item for item in first_record.resolved_request["native_stage_receipts"]
        if item["stage"] == "gate"
    )
    second_gate = next(
        item for item in second_record.resolved_request["native_stage_receipts"]
        if item["stage"] == "gate"
    )

    assert first_gate == second_gate


def test_pricing_is_published_before_gate_and_diagnostics_cannot_replace_fair_value():
    base = _planned_native()
    inputs = replace(
        base,
        gate={
            "model": {
                "intercept": 4.0,
                "coefficients": {"entry_cost": -1.0},
            },
            "threshold": 0.0,
        },
    )

    worst = application.score_one(_request(0.0), inputs)
    best = application.score_one(_request(1.0), inputs)

    assert worst.gate_terms == {
        "gate_score": -2.0,
        "gate_threshold": 0.0,
        "gate_pass": False,
    }
    assert best.gate_terms == {
        "gate_score": 2.0,
        "gate_threshold": 0.0,
        "gate_pass": True,
    }
    assert worst.financial_diagnostics["fair_premium_pct"] == pytest.approx(4.6)
    assert best.financial_diagnostics["fair_premium_pct"] == pytest.approx(4.6)


def _analog_population() -> list[dict]:
    return [
        {"row_id": "a", "features": {"move": 1.0}, "realized_pnl": 3.0},
        {"row_id": "b", "features": {"move": -1.0}, "realized_pnl": -1.0},
        {"row_id": "c", "features": {"move": 1.0}, "realized_pnl": 100.0},
    ]


def _analog_recipe(rows: list[dict]) -> dict:
    # A plain dict, matching how source_inputs._bounded_recipe builds the
    # real "analogs.recipe" block (not the AnalogRecipe dataclass, which
    # is a separate accepted shape used only by direct evaluate_analogs
    # callers).
    return {
        "feature_names": ("move",),
        "neighbors": 2,
        "population_hash": source_population_hash(rows),
    }


def test_no_analog_recipe_stays_silent_and_unowned_check_still_fires():
    # Genuinely not-applicable: analogs={} carries no recipe and no inputs at
    # all (R4-7 negative control). This must keep behaving exactly as today:
    # no MISSING_ANALOG_INPUT flag, no analog outputs, record still scores.
    record = application.score_one(_request(), _native())

    assert "MISSING_ANALOG_INPUT" not in record.reason_codes
    assert record.resolved_request.get("exp_pnl_analog") is None
    assert record.validation_status == "scored"

    # A stray owned output with no recipe/inputs at all is still reported by
    # the pre-existing UNOWNED_ANALOG_OUTPUT check, unaffected by this fix.
    stray = application.score_one(
        _request(), replace(_native(), analogs={"exp_pnl_analog": 0.1}),
    )
    assert "UNOWNED_ANALOG_OUTPUT" in stray.reason_codes


def test_analog_recipe_with_inputs_present_computes_outputs():
    rows = _analog_population()
    inputs = replace(
        _native(),
        analogs={
            "recipe": _analog_recipe(rows),
            "source_rows": rows,
            "query_features": {"move": 1.0},
        },
    )

    record = application.score_one(_request(), inputs)

    assert "MISSING_ANALOG_INPUT" not in record.reason_codes
    assert record.resolved_request["exp_pnl_analog"] is not None
    assert record.resolved_request["n_analogs"] == 2


def test_analog_recipe_with_missing_required_inputs_is_reported_not_silent():
    rows = _analog_population()
    recipe = _analog_recipe(rows)

    # Recipe present, both source_rows and query_features absent: the
    # defect this task fixes (R4-7). Previously returned {} with no flag.
    both_missing = application.score_one(
        _request(), replace(_native(), analogs={"recipe": recipe}),
    )
    assert "MISSING_ANALOG_INPUT" in both_missing.reason_codes
    assert both_missing.validation_status == "refused"
    assert both_missing.resolved_request.get("exp_pnl_analog") is None

    # Recipe present, only source_rows absent: already reported before this
    # fix (existing isinstance check) and must remain reported.
    source_missing = application.score_one(
        _request(),
        replace(
            _native(),
            analogs={"recipe": recipe, "query_features": {"move": 1.0}},
        ),
    )
    assert "MISSING_ANALOG_INPUT" in source_missing.reason_codes
    assert source_missing.resolved_request.get("exp_pnl_analog") is None

    # Recipe present, only query_features absent: already reported before
    # this fix and must remain reported.
    query_missing = application.score_one(
        _request(),
        replace(_native(), analogs={"recipe": recipe, "source_rows": rows}),
    )
    assert "MISSING_ANALOG_INPUT" in query_missing.reason_codes
    assert query_missing.resolved_request.get("exp_pnl_analog") is None
