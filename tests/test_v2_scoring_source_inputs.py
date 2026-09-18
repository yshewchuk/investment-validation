from dataclasses import replace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import (
    SourceBundle,
    build_native_score_inputs,
)


def _bundle() -> SourceBundle:
    expiry = "2026-09-18"
    return SourceBundle(
        source_ref="synthetic:str-thru:v1",
        context={
            "ticker": "TEST",
            "event_date": "2026-09-16",
            "entry_date": "2026-09-16",
            "exit_date": expiry,
            "expiry": expiry,
            "spot": 100.0,
        },
        raw_quotes={
            ("C", 100.0, expiry): {"bid": 1.0, "ask": 3.0},
            ("P", 100.0, expiry): {"bid": 1.0, "ask": 3.0},
        },
        feature_vector={"signal": 1.0},
        feature_missing_mask={"signal": False},
        model_identity={
            "driver": {
                "model_id": "synthetic-driver-v1",
                "recipe_id": "linear-v1",
            },
        },
        forecast_recipes={
            "driver_prediction": {
                "intercept": 5.0,
                "coefficients": {"signal": 2.0},
            },
        },
        model_artifact_refs={
            "driver_prediction": "sha256:synthetic-driver",
        },
        residual_recipe={
            "mode": "terminal",
            "terminal_spots": (90.0, 110.0),
            "weights": (1.0, 1.0),
            "capital_at_risk": 4.0,
            "population_ref": "synthetic:terminal-scenarios:v1",
            "recipe_id": "terminal-payoff-v1",
            "draw_count": 2,
            "seed": 17,
        },
        analog_recipe={
            "bucket_dimensions": (
                "mcap_bucket", "moneyness_band", "dte_band", "implied_tercile",
            ),
            "widening_order": ("moneyness_band", "dte_band", "implied_tercile"),
            "min_analogs": 2,
            "alpha": 0.5,
            "bootstrap_draws": 0,
            "bootstrap_seed": 0,
            "ci_quantiles": (0.1, 0.9),
        },
        analog_source_rows=(
            {"row_id": "a1", "mcap_bucket": "large", "moneyness_band": "atm",
             "dte_band": "30-45", "implied_tercile": "mid",
             "realized_return": 0.10},
            {"row_id": "a2", "mcap_bucket": "large", "moneyness_band": "atm",
             "dte_band": "30-45", "implied_tercile": "mid",
             "realized_return": 0.20},
            {"row_id": "a3", "mcap_bucket": "small", "moneyness_band": "otm",
             "dte_band": "0-15", "implied_tercile": "low",
             "realized_return": -0.50},
        ),
        analog_query={
            "mcap_bucket": "large", "moneyness_band": "atm",
            "dte_band": "30-45", "implied_tercile": "mid",
        },
        gate_recipe={
            "recipe_id": "linear-gate-v1",
            "artifact_ref": "sha256:synthetic-gate",
            "model": {
                "intercept": 4.0,
                "coefficients": {"entry_cost": -1.0},
            },
            "threshold": 0.0,
        },
        metadata={"snapshot_ref": "synthetic:snapshot:v1"},
    )


def _request(alpha: float) -> ScoreRequest:
    return ScoreRequest(
        event_id="TEST-2026-09-16",
        calendar_revision="synthetic-calendar-v1",
        strategy_version="STR-THRU",
        deployment_id="synthetic-deployment-v1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="synthetic-snapshot-v1",
        mode="replay",
        fill_model={"alpha": alpha},
        model_artifact_refs=("sha256:synthetic-driver", "sha256:synthetic-gate"),
        residual_state_ref="synthetic:terminal-scenarios:v1",
        analog_state_ref="synthetic:analogs:v1",
    )


def test_builder_rejects_answers_and_keeps_geometry_and_pricing_unresolved():
    inputs = build_native_score_inputs(_bundle())

    assert inputs.geometry is None
    assert inputs.pricing is None
    assert "frozen_outputs" not in inputs.forecast
    assert "entry_cost" not in inputs.context
    assert "selected_contracts" not in inputs.context
    assert "exp_pnl_sim" not in inputs.simulation
    assert "gate_pass" not in inputs.gate
    assert inputs.diagnostics == {}

    with pytest.raises(ValueError, match="calculated answer fields"):
        build_native_score_inputs(replace(
            _bundle(),
            context={**_bundle().context, "entry_cost": 4.0},
        ))
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        build_native_score_inputs(replace(
            _bundle(),
            analog_recipe={**_bundle().analog_recipe, "gate_pass": True},
        ))
    with pytest.raises(ValueError, match="unsupported fields"):
        build_native_score_inputs(replace(
            _bundle(),
            raw_quotes={
                ("C", 100.0, "2026-09-18"): {
                    "bid": 1.0,
                    "ask": 3.0,
                    "selected_contracts": (),
                },
            },
        ))


def test_native_scoring_runs_from_source_inputs():
    inputs = build_native_score_inputs(_bundle())
    record = application.score_one(_request(0.5), inputs)

    assert record.validation_status == "scored"
    assert record.readiness == "ready"
    # WIDE_MARKET is a real, correctly-derived flag: the fixture's
    # bid=1.0/ask=3.0 quote is genuinely wide by WIDE_MARKET_RATIO=0.5. It
    # is advisory (engine/fills.py:150 `FillModel.is_wide`; corpus evidence
    # in engine/v2/scoring/stages.py's ADVISORY_FLAGS), so it annotates
    # this row without refusing it.
    assert record.reason_codes == ("WIDE_MARKET",)
    assert record.forecasts["driver_prediction"] == pytest.approx(7.0)
    assert record.resolved_request["entry_cost"] == pytest.approx(4.0)
    assert record.forecasts["exp_pnl_sim"] == pytest.approx(1.5)
    assert record.gate_terms == {
        "gate_score": pytest.approx(0.0),
        "gate_threshold": pytest.approx(0.0),
        "gate_pass": True,
    }
    assert {(leg["right"], leg["strike"]) for leg in record.selected_contracts} == {
        ("C", 100.0),
        ("P", 100.0),
    }
    assert record.resolved_request["native_source_ref"] == _bundle().source_ref
    # Analogs must actually be computed from the source bucket population,
    # not merely pass through without raising MISSING_ANALOG_INPUT: real
    # numbers, derived from the two matching rows (0.10, 0.20), prove the
    # legacy-faithful bucket path in native_analog._evaluate_bucket_analogs
    # ran rather than being silently skipped.
    assert record.resolved_request["exp_pnl_analog"] == pytest.approx(0.15)
    assert record.resolved_request["win_analog"] == pytest.approx(1.0)
    assert record.resolved_request["n_analogs"] == 2


def test_fill_reprices_and_propagates_to_simulation_and_gate():
    inputs = build_native_score_inputs(_bundle())
    worst = application.score_one(_request(0.0), inputs)
    best = application.score_one(_request(1.0), inputs)

    assert worst.resolved_request["entry_cost"] == pytest.approx(6.0)
    assert best.resolved_request["entry_cost"] == pytest.approx(2.0)
    assert worst.forecasts["exp_pnl_sim"] == pytest.approx(1.0)
    assert best.forecasts["exp_pnl_sim"] == pytest.approx(2.0)
    assert worst.gate_terms["gate_score"] == pytest.approx(-2.0)
    assert best.gate_terms["gate_score"] == pytest.approx(2.0)
    assert worst.gate_terms["gate_pass"] is False
    assert best.gate_terms["gate_pass"] is True
    assert worst.selected_contracts == best.selected_contracts


@pytest.mark.parametrize("strategy", ["STR-RUNUP", "BFLY-P", "CTR5"])
def test_supported_source_strategies_require_their_declared_forecast_inputs(strategy):
    bundle = replace(
        _bundle(),
        strategy=strategy,
        model_artifact_refs={
            "driver_prediction": "sha256:synthetic-driver",
            **({"runup_move_prediction": "sha256:synthetic-runup"}
               if strategy == "STR-RUNUP" else
               {"forecast_abs_move": "sha256:synthetic-size"}),
        },
        forecast_recipes={
            "driver_prediction": {
                "intercept": 5.0,
                "coefficients": {"signal": 2.0},
            },
            **({"runup_move_prediction": {
                "intercept": 1.0,
                "coefficients": {"signal": 0.5},
            }} if strategy == "STR-RUNUP" else {
                "forecast_abs_move": {
                    "intercept": 1.0,
                    "coefficients": {"signal": 0.5},
                },
            }),
        },
    )
    inputs = build_native_score_inputs(bundle)

    assert inputs.context["strategy"] == strategy
    assert set(inputs.forecast["models"]) >= set(bundle.forecast_recipes)


def test_nested_answer_fields_are_rejected():
    with pytest.raises(ValueError, match="calculated answer fields"):
        build_native_score_inputs(replace(
            _bundle(),
            metadata={"recipe": {"financial_diagnostics": {"fair": 1.0}}},
        ))


def test_analog_recipe_without_source_rows_reports_missing_input():
    # A declared-but-unfed recipe must surface MISSING_ANALOG_INPUT from the
    # execution stage -- it must not be silently reclassified as
    # not-applicable just because analog_source_rows is empty.
    bundle = replace(_bundle(), analog_source_rows=())
    inputs = build_native_score_inputs(bundle)

    assert inputs.analogs["recipe"] is not None
    assert "source_rows" not in inputs.analogs
    assert "query_features" not in inputs.analogs

    record = application.score_one(_request(0.5), inputs)

    assert record.validation_status == "refused"
    assert "MISSING_ANALOG_INPUT" in record.reason_codes
