from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import Pricing, generate, price
from engine.v2.scoring import application
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _request(**changes):
    request = ScoreRequest(
        event_id="evt-canonical",
        calendar_revision="cal-1",
        strategy_version="STR-THRU",
        deployment_id="dep-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snap-1",
        mode="replay",
        fill_model={"alpha": 0.5},
    )
    return replace(request, **changes)


def _receipts():
    return tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES
        if stage != "diagnostics"
    )


def _override_inputs():
    context = {
        "ticker": "AAA",
        "strategy": "STR-THRU",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-16",
        "exit_date": "2026-09-17",
        "expiry": "2026-09-18",
        "spot": 100.0,
        "strike": 95.0,
    }
    features = {
        "strike": 100.0,
        "model_inputs": {"strike": 100.0},
    }
    forecast = {
        "models": {
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
        },
    }
    stale_geometry = generate("STR-THRU", {**context, "strike": 95.0})
    priced_legs = []
    for strike, bid, ask in (
        (95.0, 0.5, 1.5),
        (100.0, 1.0, 2.0),
        (105.0, 2.0, 4.0),
        (110.0, 3.0, 6.0),
    ):
        geometry = generate("STR-THRU", {**context, "strike": strike})
        quotes = {
            (leg.right, leg.strike, leg.expiry): {"bid": bid, "ask": ask}
            for leg in geometry.legs
        }
        priced_legs.extend(price(geometry, quotes, 0.5).legs)
    pricing = Pricing("STR-THRU", 100.0, 0.0, tuple(priced_legs))
    return NativeScoreInputs(
        context=context,
        features=features,
        forecast=forecast,
        geometry=stale_geometry,
        pricing=pricing,
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="typed-native-fixture",
        stage_receipts=_receipts(),
    )


@pytest.mark.parametrize(
    ("request_changes", "expected_strike", "expected_cost"),
    (
        ({"geometry_override": {"strike": 105.0}}, 105.0, 6.0),
        ({"contract_override": {"strike": 110.0}}, 110.0, 9.0),
    ),
)
def test_request_overrides_beat_context_features_and_reprice(
    request_changes, expected_strike, expected_cost,
):
    record = application.score_one(
        _request(**request_changes),
        _override_inputs(),
    )

    assert {leg["strike"] for leg in record.selected_contracts} == {
        expected_strike,
    }
    assert {leg["strike"] for leg in record.legs} == {expected_strike}
    assert record.resolved_request["entry_cost"] == pytest.approx(expected_cost)
    assert record.financial_diagnostics["entry_cost_pct"] == pytest.approx(
        expected_cost,
    )


def _frozen_fields():
    base = _override_inputs()
    return {
        "_native_inputs": replace(
            base,
            geometry=None,
            forecast={
                "driver_name": "abs_move",
                "models": {
                    "driver_prediction": {
                        "intercept": 99.0,
                        "coefficients": {},
                    },
                    "forecast_abs_move": {
                        "intercept": 88.0,
                        "coefficients": {},
                    },
                },
            },
        ),
    }


def _release():
    return SimpleNamespace(
        release_id="release-1",
        bindings=(
            SimpleNamespace(
                binding_id="driver-binding",
                role="implied_t1",
                output_names=("prediction",),
            ),
            SimpleNamespace(
                binding_id="size-binding",
                role="size",
                output_names=("prediction",),
            ),
        ),
    )


def _inference_requests():
    return (
        SimpleNamespace(binding_id="driver-binding"),
        SimpleNamespace(binding_id="size-binding"),
    )


class _Frozen:
    def infer(self, release, inference_request):
        values = {
            "driver-binding": 0.77,
            "size-binding": 0.42,
        }
        return SimpleNamespace(
            status="READY",
            release_id=release.release_id,
            binding_id=inference_request.binding_id,
            model_id=f"model-{inference_request.binding_id}",
            output_names=("prediction",),
            predictions=((values[inference_request.binding_id],),),
            artifact_hashes=(f"sha256:{inference_request.binding_id}",),
            reason_codes=(),
        )


def test_score_frozen_preserves_role_outputs_over_local_recipes():
    record = application.score_frozen(
        _request(),
        _Frozen(),
        _release(),
        _inference_requests(),
        _frozen_fields(),
    )

    assert record.forecasts["driver_prediction"] == pytest.approx(0.77)
    assert record.forecasts["forecast_abs_move"] == pytest.approx(0.42)
    assert record.model_artifact_ids == (
        "sha256:driver-binding",
        "sha256:size-binding",
    )


def test_str_thru_frozen_driver_role_publishes_driver_prediction():
    release = SimpleNamespace(
        release_id="release-1",
        bindings=(SimpleNamespace(
            binding_id="driver-binding",
            role="driver",
            output_names=("driver_prediction",),
        ),),
    )

    fields = _frozen_fields()
    fields["_native_inputs"] = replace(
        fields["_native_inputs"],
        forecast={"required_roles": ("driver",)},
        simulation={"mode": "not_applicable"},
    )
    record = application.score_frozen(
        _request(),
        _Frozen(),
        release,
        (SimpleNamespace(binding_id="driver-binding"),),
        fields,
    )

    assert record.forecasts["driver_prediction"] == pytest.approx(0.77)
    assert record.forecasts["forecast_abs_move"] is None
    assert "MISSING_FORECAST_OUTPUT:driver" not in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:size" not in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:iv_crush" not in record.reason_codes


def test_score_frozen_artifact_mismatch_refuses_without_local_fallback():
    class ArtifactMismatch(_Frozen):
        def infer(self, release, inference_request):
            if inference_request.binding_id == "driver-binding":
                return SimpleNamespace(
                    status="MODEL_NOT_READY",
                    release_id=release.release_id,
                    binding_id=inference_request.binding_id,
                    model_id="model-driver-binding",
                    output_names=("prediction",),
                    predictions=(),
                    artifact_hashes=("sha256:driver-binding",),
                    reason_codes=("ARTIFACT_INVALID",),
                )
            return super().infer(release, inference_request)

    record = application.score_frozen(
        _request(),
        ArtifactMismatch(),
        _release(),
        _inference_requests(),
        _frozen_fields(),
    )

    assert record.forecasts["driver_prediction"] is None
    assert record.forecasts["forecast_abs_move"] == pytest.approx(0.42)
    assert "ARTIFACT_INVALID" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:implied_t1" in record.reason_codes
    assert record.validation_status == "refused"
    assert record.readiness == "refused"


def test_typed_frozen_gate_uses_calculated_price_and_existing_threshold():
    member = SimpleNamespace(name="model", content_hash="sha256:gate")
    binding = SimpleNamespace(
        binding_id="gate-binding",
        model_id="gate-model",
        role="gate",
        feature_order=("entry_cost",),
        output_names=("prediction",),
        members=(member,),
    )
    release = SimpleNamespace(release_id="release-1", bindings=(binding,))
    inference_request = SimpleNamespace(
        binding_id="gate-binding",
        rows=((999.0,),),
    )

    class GateInference:
        def infer(self, model_release, request):
            return SimpleNamespace(
                status="READY",
                release_id=model_release.release_id,
                binding_id=request.binding_id,
                model_id="gate-model",
                output_names=("prediction",),
                predictions=((request.rows[0][0],),),
                artifact_hashes=("sha256:gate",),
                reason_codes=(),
                detail=None,
            )

    base = _override_inputs()
    record = application.score_frozen(
        _request(),
        GateInference(),
        release,
        inference_request,
        {
            "_native_inputs": replace(
                base,
                gate={"threshold": 1.0},
            ),
        },
    )

    assert record.gate_terms["gate_score"] == pytest.approx(
        record.resolved_request["entry_cost"],
    )
    assert record.gate_terms["gate_score"] != pytest.approx(999.0)
    assert record.gate_terms["gate_threshold"] == pytest.approx(1.0)


def test_typed_frozen_gate_ignores_caller_supplied_gate_threshold():
    """gate_threshold is answer-bearing (source_inputs._ANSWER_FIELDS); the
    frozen path must take it only from the native bundle's own gate block,
    never from a caller-supplied fields mapping passed alongside it."""
    member = SimpleNamespace(name="model", content_hash="sha256:gate")
    binding = SimpleNamespace(
        binding_id="gate-binding",
        model_id="gate-model",
        role="gate",
        feature_order=("entry_cost",),
        output_names=("prediction",),
        members=(member,),
    )
    release = SimpleNamespace(release_id="release-1", bindings=(binding,))
    inference_request = SimpleNamespace(
        binding_id="gate-binding",
        rows=((999.0,),),
    )

    class GateInference:
        def infer(self, model_release, request):
            return SimpleNamespace(
                status="READY",
                release_id=model_release.release_id,
                binding_id=request.binding_id,
                model_id="gate-model",
                output_names=("prediction",),
                predictions=((request.rows[0][0],),),
                artifact_hashes=("sha256:gate",),
                reason_codes=(),
                detail=None,
            )

    base = _override_inputs()
    fields = {
        "_native_inputs": replace(base, gate={"threshold": 1.0}),
        "gate_threshold": 999.0,
    }
    record = application.score_frozen(
        _request(), GateInference(), release, inference_request, fields,
    )

    assert record.gate_terms["gate_threshold"] == pytest.approx(1.0)
