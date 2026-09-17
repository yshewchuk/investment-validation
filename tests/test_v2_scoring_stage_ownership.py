import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.scoring import application
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
